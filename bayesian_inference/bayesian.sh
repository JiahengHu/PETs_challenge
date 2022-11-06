#!/usr/bin/env bash

DISEASE=../sample_data/va_disease_outcome_training.csv
POP=../sample_data/va_person.csv
GRAPH=../sample_data/va_population_network.csv


python make_bayesian_data.py $GRAPH $DISEASE $POP


