//
// Created by orenaud on 4/14/25.
//
#include "casacore_wrapper.h"
#include <iostream>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/OS/File.h>
#include <fftw3.h>
#include <cmath>

#define SPEED_OF_LIGHT 299792458.0

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif



extern "C" void load_visibilities_from_ms(const char* ms_path_c, int num_vis,
                                          Config* config, Visibility* uvw_coords, Complex* measured_vis)
{
    std::string ms_path(ms_path_c);
    std::cout << "UPDATE >>> Loading visibilities from MS file " << ms_path << "...\n";

    if (!casacore::File(ms_path).exists()) {
        std::cerr << "ERROR >>> MeasurementSet " << ms_path << " does not exist!\n";
        std::exit(EXIT_FAILURE);
    }

    casacore::MeasurementSet ms(ms_path);
    casacore::MSColumns msCols(ms);

    int total_rows = ms.nrow();
    if (num_vis > total_rows) {
        std::cerr << "ERROR >>> Requested more visibilities than available rows.\n";
        std::exit(EXIT_FAILURE);
    }

    casacore::ArrayColumn<casacore::Double> uvwCol(ms, "UVW");
    size_t nrows = ms.nrow();

    std::cout << "Nombre de visibilités : " << nrows << std::endl;
    std::cout << "🌌 Coordonnées UVW des 5 premières lignes :" << std::endl;

    for (size_t i = 0; i < std::min(nrows, size_t(5)); ++i) {
        casacore::Vector<casacore::Double> uvw;
        uvwCol.get(i, uvw);
        std::cout << "Visibilité " << i << " : "
                  << "u = " << uvw[0] << " m, "
                  << "v = " << uvw[1] << " m, "
                  << "w = " << uvw[2] << " m" << std::endl;
    }

    auto uvw_col = msCols.uvw().getColumn();
    auto data_col = msCols.data().getColumn();
    auto weight_col = msCols.weight().getColumn();

    auto uvw_vec = uvw_col.tovector();
    auto data_vec = data_col.tovector();
    auto weight_vec = weight_col.tovector();

    auto shape = data_col.shape();
    std::cout << "Shape of DATA cell: [";
    for (size_t i = 0; i < shape.size(); ++i)
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    std::cout << "]\n";

    int nchan = shape[0];
    int npol = shape[1];

    double maxW = -std::numeric_limits<double>::infinity();


    for (int i = 0; i < num_vis; ++i)
    {
        casacore::Vector<casacore::Double> uvw;
        uvwCol.get(i, uvw);
        double u = uvw[0];
        double v = uvw[1];
        double w = uvw[2];

        if (w > maxW) {
            maxW = w;
        }

        if (config->right_ascension) {
            u *= -1.0;
            w *= -1.0;
        }

        uvw_coords[i].u = u ;
        uvw_coords[i].v = v ;
        uvw_coords[i].w = w ;

        auto cpx = data_vec[i * nchan * npol];  // premier canal/pola
        float weight = config->force_weight_to_one ? 1.0f : weight_vec[i];

        measured_vis[i].real = cpx.real() * weight;
        measured_vis[i].imaginary = cpx.imag() * weight;
    }
    std::cout << "UPDATE >>> Loaded " << num_vis << " visibilities from MeasurementSet.\n";

    config->max_w = maxW;
    int NUM_KERNELS = 17;
    config->w_scale = pow(NUM_KERNELS - 1, 2.0) / config->max_w;

    casacore::MSSpectralWindow spwTable = ms.spectralWindow();
    casacore::ArrayColumn<casacore::Double> chanFreqCol(spwTable, "CHAN_FREQ");

    // Récupère la fréquence centrale (première valeur du tableau)
    casacore::Array<casacore::Double> freqs;
    chanFreqCol.get(0, freqs);
    double freq_hz = freqs(casacore::IPosition(1, 0));
    config->frequency_hz = freq_hz;

    // Sous-table ANTENNA
    casacore::MSAntenna antTable = ms.antenna();
    casacore::ScalarColumn<casacore::Double> dishDiameterCol(antTable, "DISH_DIAMETER");

    // Récupère le diamètre de la première antenne
    double D = dishDiameterCol(0); // en mètres

    // Calcule la longueur d'onde
    double wavelength = SPEED_OF_LIGHT / freq_hz;

    // Calcule le champ de vue (en radians puis en degrés)
    double fov_rad = 1.22 * wavelength / D;
    double fov_deg = fov_rad * 180.0 / M_PI;
    std::cout << "📡 FoV ≈ " << fov_deg << " degrés" << std::endl;
    int GRID_SIZE = 1024;
    config->cell_size = (fov_deg * PI) / (180.0 * GRID_SIZE);
    config->uv_scale =  config->cell_size*GRID_SIZE;
}

extern "C" void psf_host_set_up_ms(int GRID_SIZE, int PSF_GRID_SIZE, Config *config, PRECISION *psf, double *psf_max_value) {
    // Ouvrir le MS
    casacore::MeasurementSet ms(config->visibility_source_file);
    casacore::ArrayColumn<casacore::Double> uvwCol(ms, "UVW");

    int nrow = ms.nrow();
    std::vector<std::pair<double, double>> uv_coords;

    // Lire les (u,v) uniquement
    for (int i = 0; i < nrow; ++i) {
        casacore::Vector<casacore::Double> uvw = uvwCol(i);
        double u = uvw[0];
        double v = uvw[1];
        uv_coords.emplace_back(u, v);
    }

    std::cout << "✅ Lu " << uv_coords.size() << " visibilités\n";

    // Créer une grille UV vide
    fftw_complex* uv_grid = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * GRID_SIZE * GRID_SIZE);
    memset(uv_grid, 0, sizeof(fftw_complex) * GRID_SIZE * GRID_SIZE);

    // Mettre à l'échelle les coordonnées UV vers la grille
    double max_uv = 0.0;
    for (const auto& [u, v] : uv_coords) {
        max_uv = std::max(max_uv, std::sqrt(u*u + v*v));
    }

    for (const auto& [u, v] : uv_coords) {
        int x = static_cast<int>((u / max_uv * 0.5 + 0.5) * (GRID_SIZE - 1));
        int y = static_cast<int>((v / max_uv * 0.5 + 0.5) * (GRID_SIZE - 1));
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            uv_grid[y * GRID_SIZE + x][0] += 1.0; // real
            uv_grid[y * GRID_SIZE + x][1] += 0.0; // imag
        }
    }

    std::cout << "✅ Grille UV construite\n";

    // Allouer la sortie PSF (dans le domaine image)
    fftw_complex* psf_grid = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * GRID_SIZE * GRID_SIZE);

    // Plan FFT
    fftw_plan plan = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, uv_grid, psf_grid, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_free(uv_grid);

    // Extraire la magnitude au centre du PSF
    int half = GRID_SIZE / 2;
    *psf_max_value = 0.0;

    for (int j = 0; j < PSF_GRID_SIZE; ++j) {
        for (int i = 0; i < PSF_GRID_SIZE; ++i) {
            int x = (half - PSF_GRID_SIZE / 2 + i + GRID_SIZE) % GRID_SIZE;
            int y = (half - PSF_GRID_SIZE / 2 + j + GRID_SIZE) % GRID_SIZE;
            int index = y * GRID_SIZE + x;

            double val = std::sqrt(psf_grid[index][0]*psf_grid[index][0] + psf_grid[index][1]*psf_grid[index][1]);
            psf[j * PSF_GRID_SIZE + i] = static_cast<PRECISION>(val);
            if (val > *psf_max_value) {
                *psf_max_value = val;
            }
        }
    }

    fftw_free(psf_grid);

    std::cout << "✅ PSF calculée. Valeur max : " << *psf_max_value << "\n";

}