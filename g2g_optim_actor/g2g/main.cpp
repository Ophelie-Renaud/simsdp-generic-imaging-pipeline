#include <iostream>
#include "libcpu_skytosky_single.h"

void initialize_interpolation_parameters(interpolation_parameters *params) {
    // Initialisation des champs
    params->nrows = 100;
    params->Nx = 512;
    params->half_Nx = 256;
    params->Ny = 512;
    params->half_Ny = 256;
    params->Nchan = 10;
    params->spw_selected = 5;
    params->oversampling_factor = 2;
    params->half_support_function = 8;
    params->full_support_function = 16;
    params->nb_w_planes = 3;
    params->nb_vis_polarization = 2;
    params->nb_grid_polarization = 2;
    params->grid_channel_idx = (size_t*)malloc(sizeof(size_t) * params->Nchan);; // Exemple d'allocation pour un champ de type c_void_p
    params->no_grid_index = 1;
    params->grid_channel_width = 10;
    params->no_chan_spw = 50;
    params->nb_grid_chan = 10;
    params->polarization_step = 1;
    params->cell_size_l = 0.5;
    params->cell_size_m = 0.5;
    params->u_scale = (float*)malloc(sizeof(float)); // Exemple d'un champ de type void*
    *params->u_scale = 1280.0f;
    params->v_scale = (float*)malloc(sizeof(float));
    *params->v_scale = 1280.0f;
    params->do_s2s = 1;
    params->len_s2s_coo = 0;
    params->a_coo = 0;
    params->u_coo = 0;
    params->v_coo = 0;
    params->ch_coo = 0;
    params->uvach_coo = 0;
    params->tot_size_coo = 0;
    params->chan_wavelength = (float*)malloc(sizeof(float) * 10);
    params->input_grid = (float*)malloc(sizeof(float) * 100);
    params->output_grid = (float*)malloc(sizeof(float) * 100);
    params->psf_grid = (float*)malloc(sizeof(float) * 100);
    params->visibilities = (float*)malloc(sizeof(float) * 10);
    params->visibility_weight = (float*)malloc(sizeof(float) * 10);
    params->uvw_coordinates = (float*)malloc(sizeof(float) * 10);
    params->gridding_conv_function = (float*)malloc(sizeof(float) * 10);
    params->filter_size = 0;
    params->filter_AA_2D = (float*)malloc(sizeof(float) * 5);
    params->filter_AA_2D_size = 5;
    params->filter_choice = 1;
    params->max_w = 100.0f;
    params->conv_norm_weight = (float*)malloc(sizeof(float) * 5);
}
void s2s_actor(int NUM_SAMPLES, int GRID_SIZE, int NUM_VISIBILITIES,interpolation_parameters params) {
    initialize_interpolation_parameters(&params);
    s2s_single_pola(params);
}


int main() {


    struct interpolation_parameters params;

s2s_actor(10,10,10,params);


    // Vérifie si output_grid est initialisé
    if (params.output_grid != nullptr) {
        // Cast vers float* si output_grid est censé être un tableau de floats
        float* grid = static_cast<float*>(params.output_grid);

        // Afficher quelques valeurs pour déboguer
        int num_elements = 10;  // Remplacer par la taille réelle
        std::cout << "Contenu de output_grid (quelques valeurs) : " << std::endl;
        for (int i = 0; i < num_elements; ++i) {
            std::cout << "grid[" << i << "] = " << grid[i] << std::endl;
        }
    } else {
        std::cout << "output_grid n'est pas initialisé." << std::endl;
    }


    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.