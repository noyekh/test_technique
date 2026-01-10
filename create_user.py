#!/usr/bin/env python3
"""
Script pour créer un nouvel utilisateur.

Usage:
    python create_user.py

Le script génère le hash bcrypt et propose d'ajouter
automatiquement l'utilisateur dans .streamlit/secrets.toml
"""

import re
import sys
from pathlib import Path

try:
    import bcrypt
except ImportError:
    print("Erreur: bcrypt non installé.")
    print("Exécuter: pip install bcrypt")
    sys.exit(1)


SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"


def validate_username(username: str) -> bool:
    """Username: lettres, chiffres, underscores uniquement."""
    return bool(re.match(r"^[a-z][a-z0-9_]{2,19}$", username))


def generate_toml_block(username: str, display_name: str, password: str) -> str:
    """Génère le bloc TOML pour un utilisateur."""
    hash_pwd = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return f'''
[auth.credentials.usernames.{username}]
name = "{display_name}"
password = "{hash_pwd}"
'''


def user_exists(username: str) -> bool:
    """Vérifie si l'utilisateur existe déjà."""
    if not SECRETS_PATH.exists():
        return False
    content = SECRETS_PATH.read_text()
    return f"[auth.credentials.usernames.{username}]" in content


def append_to_secrets(block: str) -> bool:
    """Ajoute le bloc au fichier secrets.toml."""
    try:
        with open(SECRETS_PATH, "a", encoding="utf-8") as f:
            f.write(block)
        return True
    except Exception as e:
        print(f"Erreur écriture: {e}")
        return False


def main():
    print("=" * 50)
    print("   CRÉATION D'UN UTILISATEUR")
    print("=" * 50)
    print()

    # Username
    while True:
        username = input("Username (ex: marie): ").strip().lower()
        if not username:
            print("Username requis.")
            continue
        if not validate_username(username):
            print("Format invalide. Règles:")
            print("  - Commence par une lettre")
            print("  - 3-20 caractères")
            print("  - Lettres minuscules, chiffres, underscores")
            continue
        if user_exists(username):
            print(f"L'utilisateur '{username}' existe déjà.")
            continue
        break

    # Display name
    display_name = input("Nom affiché (ex: Marie Dupont): ").strip()
    if not display_name:
        display_name = username.title()

    # Password
    while True:
        password = input("Mot de passe (min 6 chars): ").strip()
        if len(password) < 6:
            print("Mot de passe trop court (minimum 6 caractères).")
            continue
        password2 = input("Confirmer le mot de passe: ").strip()
        if password != password2:
            print("Les mots de passe ne correspondent pas.")
            continue
        break

    # Generate block
    print()
    print("-" * 50)
    block = generate_toml_block(username, display_name, password)
    print("Bloc TOML généré:")
    print(block)
    print("-" * 50)

    # Propose to append
    if SECRETS_PATH.exists():
        choice = input(f"Ajouter à {SECRETS_PATH.name}? [O/n]: ").strip().lower()
        if choice in ("", "o", "oui", "y", "yes"):
            if append_to_secrets(block):
                print()
                print(f"Utilisateur '{username}' ajouté.")
                print(f"Connexion: {username} / {password}")
            else:
                print("Échec. Copier le bloc manuellement.")
        else:
            print("Copier le bloc ci-dessus dans .streamlit/secrets.toml")
    else:
        print(f"Fichier {SECRETS_PATH} introuvable.")
        print("Créer le fichier et copier le bloc ci-dessus.")


if __name__ == "__main__":
    main()
