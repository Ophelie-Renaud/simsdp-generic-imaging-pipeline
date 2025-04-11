// ms_reader.cpp
#include <casacore/tables/Tables.h>
#include <casacore/casa/Arrays/Cube.h> // Ajouter cet en-tête
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Array.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <cstdlib>

extern "C" {
    void read_ms(const char* filename) {
        /*try {
            casacore::Table ms(filename, casacore::Table::Old);
            std::cout << "MS chargé : " << filename << std::endl;
        } catch (std::exception& e) {
            std::cerr << "Erreur : " << e.what() << std::endl;
        }*/
        try {
            casacore::Table ms(filename, casacore::Table::Old);
            std::cout << "MS chargé : " << filename << std::endl;

            // Extraire les colonnes UVW et DATA
            casacore::ArrayColumn<casacore::Double> uCol(ms, "UVW");
            casacore::ArrayColumn<casacore::Complex> visCol(ms, "DATA");

            std::ofstream outfile("uv_data.txt");

            casacore::uInt nrows = ms.nrow();
            std::cout << "Nombre de row : " << nrows << std::endl;
            for (casacore::uInt i = 0; i < nrows; i++) {
                // Lire u, v, w et la visibilité (complexe)
                casacore::Array<double> uv = uCol(i);
                //std::cout << "Taille de uv : " << uv.shape() << std::endl;

                casacore::Array<casacore::Complex> visArray = visCol(i);
                //std::cout << "Taille de visArray : " << visArray.shape() << std::endl;

                if (visArray.shape().product() > 0) {  // Vérifie qu'il y a bien un élément
                    casacore::Complex vis = visArray(casacore::IPosition(2, 0, 0));

                    // Accès aux coordonnées UV
                    casacore::IPosition pos(1, 0);
                    double u = uv(pos);
                    pos(0) = 1;
                    double v = uv(pos);

                    double amplitude = std::abs(vis);

                    // Écriture des données
                    outfile << u << " " << v << " " << amplitude << std::endl;
                } else {
                    std::cerr << "Attention : visArray vide à la ligne " << i << std::endl;
                }
            }

            outfile.close();
            std::cout << "Données enregistrées dans uv_data.txt" << std::endl;

            // Exécuter Gnuplot pour afficher les données
            std::cout << "Affichage des données avec Gnuplot..." << std::endl;
            system("gnuplot -persist -e \"set title 'UV Coverage'; set xlabel 'u'; set ylabel 'v'; plot 'uv_data.txt' using 1:2 with points title 'Visibilities'\"");
        } catch (std::exception& e) {
            std::cerr << "Erreur : " << e.what() << std::endl;
        }
    }
}

