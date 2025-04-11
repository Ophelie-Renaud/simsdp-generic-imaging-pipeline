# Parameterized multithreaded code

Les dossiers ci-contre contiennent les code généré par preesm puis ajuster manuellement par @orenaud pour permettre une exécution paramétrique. A savoir que le code a été générer avec la valeur des paramètre maximal pour chaque range de sorte à permettre l’exécution paramétrique sans overlap mémoire. C'est sûrement pas le meilleur choix en terme de gestion mémoire mais facilite la mise en place du benchmark.

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

Si tu veux exécuter les pipeline sur une range de paramètres:

```bash
./run_experiment g2g #y'a le choix avec g2g, dft, fft, g2g_clean
./run_experiment all # execute tout le monde
```
Chaque exécution est sauvegardé dans un fichier log (ex:`g2g.csv`) dans le dossier parent (ex: `code_g2g`).

