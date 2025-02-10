import os
import subprocess
import sys

# Configuration
GRID5000_USER = "votre_nom_utilisateur"  # Remplacez par votre nom d'utilisateur Grid5000
GRID5000_SITE = "grenoble"  # Remplacez par votre site (ex : grenoble, lille, etc.)
REMOTE_DIR = "/home/{}/scripts".format(GRID5000_USER)  # Dossier distant où placer vos scripts
LOCAL_DIR = "chemin/vers/votre/dossier_local"  # Dossier local contenant vos scripts
JOB_SCRIPT = "run_job.sh"  # Nom du script de soumission SLURM ou OAR

# Commande pour transférer le dossier via scp
def transfer_scripts():
    try:
        scp_command = [
            "scp",
            "-r",
            LOCAL_DIR,
            f"{GRID5000_USER}@access.{GRID5000_SITE}.grid5000.fr:{REMOTE_DIR}"
        ]
        print(f"Transfert des scripts avec : {' '.join(scp_command)}")
        subprocess.check_call(scp_command)
    except subprocess.CalledProcessError as e:
        print("Erreur lors du transfert des scripts :", e)
        sys.exit(1)

# Commande pour soumettre le job sur Grid5000
def submit_job():
    try:
        ssh_command = [
            "ssh",
            f"{GRID5000_USER}@access.{GRID5000_SITE}.grid5000.fr",
            f"oarsub -l nodes=1,walltime=00:30:00 -t deploy --project my_project '{REMOTE_DIR}/{JOB_SCRIPT}'"
        ]
        print(f"Soumission du job avec : {' '.join(ssh_command)}")
        subprocess.check_call(ssh_command)
    except subprocess.CalledProcessError as e:
        print("Erreur lors de la soumission du job :", e)
        sys.exit(1)

if __name__ == "__main__":
    print("Transfert des scripts sur Grid5000...")
    transfer_scripts()
    print("Soumission du job...")
    submit_job()
    print("Scripts transférés et job soumis avec succès.")

