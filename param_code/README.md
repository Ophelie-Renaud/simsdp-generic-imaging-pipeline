# Parameterized multithreaded code

Les dossiers ci-contre contiennent les code generé par preesm puis ajuster manuellement par @orenaud pour permetre une execution paramétrique. A savoir que le code a été générer avec la valeur des paramètre mximal pour chaque range de sorte à permettre l'execution paramétrique sans overlap memoire. C'est surment pas le meilleur choix en terme de gestion memoire mais facilite la mise en place du benchmark.

Si tu veux executer pipeline par pipeline:

```bash
cd code/

rm "CMakeCache.txt"
cmake .
make

./SEP_Pipeline <NUM_VIS> <GRID_SIZE> <NUM_MINOR_CYCLE>
```
- `NUM_VIS` max = 3924480

- `GRID_SIZE` max = 2560
- `NUM_MINOR_CYCLE` max = `250`

Si tu executer les pipeline sur une range de parametre:

```bash
./run_experiment g2g #y'a le choix avec g2g, dft, fft, g2g_clean
./run_experiment all # execute tout le monde
```
Chaque execution est sauvegardé dans un fichier log `g2g.csv`.

