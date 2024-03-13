#!/bin/bash

# Define the range of p and q values you want to iterate through
p_values=(0.3)
q_values=(0.6 0.7 0.8 0.9 1)

# Loop through all combinations of p and q values
for p in "${p_values[@]}"; do
  for q in "${q_values[@]}"; do
    # Run the commands with the current p and q values
    echo $p $q
    python3 ../encoder.py --opponent ../data/football/test-1.txt --p $p --q $q > football_mdp.txt
    echo "encoder done"
    python3 ../planner.py --mdp football_mdp.txt > value.txt
    python3 ../decoder.py --value-policy value.txt --opponent ../data/football/test-1.txt > policyfile.txt
    
    # Store the last policyfile.txt for this combination
    last_policyfile="policyfile_${p}_${q}.txt"
    cp policyfile.txt $last_policyfile
  done
done
