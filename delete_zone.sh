#!/bin/bash

# Supprimer récursivement les fichiers contenant :Zone.Identifier dans leur nom
find . -type f -name '*:Zone.Identifier' -print -delete

echo "Suppression terminée des fichiers :Zone.Identifier"