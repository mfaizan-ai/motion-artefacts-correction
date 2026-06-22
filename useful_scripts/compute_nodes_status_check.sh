#!/bin/bash

echo "=== GPU partition nodes ==="
sinfo -p gpu -o "%20N %10t %20E %R"

echo
echo "=== Node problems / reasons ==="
sinfo -R

echo
echo "=== Reservations ==="
scontrol show reservation

echo
echo "=== Your pending jobs ==="
squeue -u $USER -o "%.18i %.10P %.20j %.8T %.10M %.6D %R"