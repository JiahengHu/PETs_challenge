"""
This script prepares data such that it suits bayesian inference pipeline

Note that this does not produce labels for evaluation, as
training and evaluation are contained in separate files
"""

import os
from math import ceil
from argparse import ArgumentParser
from glob import glob
from pickle import load

import pandas as pd
import numpy as np
import csv

from igraph import Graph

from make_graph import make_graph


def make_si_table(disease_data):
    """
    make table of all infections indexed by pid and infected (day of infection)
    """
    is_infected = disease_data["state"] == "I"
    pid = disease_data[is_infected]["pid"]
    day = disease_data[is_infected]["day"]

    inf_time_df = pd.DataFrame({"pid" : pid, "day" : day})

    is_rec = disease_data["state"] == "R"
    pid = disease_data[is_rec]["pid"]
    day = disease_data[is_rec]["day"]

    rec_time_df = pd.DataFrame({"pid" : pid, "day" : day})

    def lookup_rec(row):
        pid_subset = rec_time_df[(rec_time_df["pid"] == row["pid"]) & (rec_time_df["day"] >= row["day"])]
        return pid_subset["day"].min()

    recovery_times = inf_time_df.apply(lookup_rec, axis=1)
    si_table = inf_time_df
    si_table.rename({"day" : "infected"}, axis=1, inplace=True)
    si_table["recovery"] = recovery_times 

    #si_table.set_index(["pid", "infected"], verify_integrity=True, inplace=True)

    return si_table



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--graph_file", help="edges csv object", default="../sample_data/va_population_network.csv")
    parser.add_argument("--disease_file", help="disease data", default="../sample_data/va_disease_outcome_training.csv")
    parser.add_argument("--pop_file", help="population characteristics file", default="../sample_data/va_person.csv")
    # parser.add_argument("out_file", help="output file for data")
    parser.add_argument("--min-date", default=0, type=int, help="Min date to generate data from (used for making evaluation set)")
    parser.add_argument("--is-eval", action="store_true",default=False, help="exclude positive instances (assume input file is training)") 
    parser.add_argument("--past-window", default=3, help="How much history to consider for each data point")
    # parser.add_argument("--pid_partition", type=int, help="index of pids partition")
    # parser.add_argument("--n_jobs", type=int, help="total number of jobs")

    args = parser.parse_args()

    pop = pd.read_csv(args.pop_file)
    disease_data = pd.read_csv(args.disease_file)

    num_of_training_days = 56
    last_day_df = disease_data.loc[disease_data["day"] == num_of_training_days]
    pop_size = len(last_day_df)
    inf_pop = len(last_day_df.loc[last_day_df["state"] != "S"])

    # ID translation tables
    pid_to_idx = dict()
    idx_to_pid = np.zeros(pop_size, dtype=int)
    for index, row in pop.iterrows():
        pid_to_idx[row['pid']] = index
        idx_to_pid[index] = row['pid']

    si_data = make_si_table(disease_data)
    median_sickness_length = np.nanmedian((si_data['recovery'] - si_data['infected']).to_numpy())

    transmission_rate =0# TODO
    decay = 0.5**(1/median_sickness_length)
    print(transmission_rate, decay)

    probs = np.zeros(pop_size, 'float32')   # * 1 / pop_size    We know who starts the outbreak
    contact_matrix2 = np.zeros((pop_size, pop_size), 'float32')
    contact_matrix3 = np.copy(contact_matrix2)

    # Compute recovery rate / transmission rate
    # contacts = pd.read_csv(args.graph_file, chunksize=1024)  # Assumes sorted - it's not going to be
    # day_start = 0
    # day = 0
    # for chunk in contacts:
    #     for index, row in chunk.iterrows():
    #         if row['start_time'] > day_start + (24 * 60):  # Day advanced
    #             day += 1
    #             day_start = row['start_time']




    # Training data
    contacts = pd.read_csv(args.graph_file, chunksize=1024)     # Assumes sorted - it's not going to be
    day_start = 0
    day = 0
    for chunk in contacts:
        for index, row in chunk.iterrows():
            if row['start_time'] > day_start + (24 * 60):   # Day advanced
                # Record ground truth infection events
                last_day_df = disease_data.loc[disease_data["day"] == day]
                infected = last_day_df.loc[last_day_df["state"] == "I"]["pid"].to_numpy()
                for i in range(len(infected)):
                    infected_pid = pid_to_idx[infected[i]]
                    probs[infected_pid] = 1.0
                # Similarly record recovered
                recovered = last_day_df.loc[last_day_df["state"] == "R"]["pid"].to_numpy()
                for i in range(len(recovered)):
                    infected_pid = pid_to_idx[recovered[i]]
                    probs[infected_pid] = 0.0

                # Update infection probabilities
                np.fill_diagonal(contact_matrix2, 1)
                np.fill_diagonal(contact_matrix3, 0)
                new_probs = np.zeros(pop_size, 'float32')
                for i in range(pop_size):
                    o1 = np.multiply(probs[i], contact_matrix3[i])
                    o2 = contact_matrix2[i] - o1
                    o3 = np.multiply(o2, probs)
                    o4 = np.subtract([1.], o3)
                    prod = o4.prod()
                    new_probs[i] = 1 - prod
                probs = new_probs
                probs *= decay
                contact_matrix2 *= (decay * decay)
                contact_matrix3 *= (decay * decay * decay)

                # Advance day counter
                day += 1
                day_start = row['start_time']
                print(day, len(infected), len(recovered))
                print('contacts', np.min(contact_matrix2), np.max(contact_matrix2), np.mean(contact_matrix2))
                print('probs', np.min(probs), np.max(probs), np.mean(probs))
                candidates = np.argpartition(probs, -8)[-8:].flatten().tolist()
                top_inf = dict()
                for cand in candidates:
                    top_inf[idx_to_pid[cand]] = probs[cand]
                print(top_inf)

            # Update contacts
            pid = pid_to_idx[row['pid1']]
            cid = pid_to_idx[row['pid2']]
            duration = row['duration'] / (60 * 24)  # Duration is in min, normalize to contact level over a day
            contact_increase = 1 - ((1 - transmission_rate) ** duration)
            inc2 = min(contact_matrix2[pid][cid] + contact_increase, 1)
            inc3 = min(contact_matrix3[cid][pid] + contact_increase, 1)
            contact_matrix2[pid][cid] = inc2
            contact_matrix2[cid][pid] = inc2
            contact_matrix3[pid][cid] = inc3
            contact_matrix3[cid][pid] = inc3

    # Replay last 7 days of contact for evaluation
    contacts = pd.read_csv(args.graph_file, chunksize=1024)
    day_start = 0
    day = 0
    for chunk in contacts:
        for index, row in chunk.iterrows():
            if row['start_time'] > day_start + (24 * 60):  # Day advanced
                # Update infection probabilities
                if day >= 50:
                    np.fill_diagonal(contact_matrix2, 1)
                    np.fill_diagonal(contact_matrix3, 0)
                    new_probs = np.zeros(pop_size, 'float32')
                    for i in range(pop_size):
                        o1 = np.multiply(probs[i], contact_matrix3[i])
                        o2 = contact_matrix2[i] - o1
                        o3 = np.multiply(o2, probs)
                        o4 = np.subtract([1.], o3)
                        prod = o4.prod()
                        new_probs[i] = 1 - prod
                    probs = new_probs
                    probs *= decay
                    contact_matrix2 *= (decay * decay)
                    contact_matrix3 *= (decay * decay * decay)

                # Advance day counter
                day += 1
                day_start = row['start_time']
                if day >= 50:
                    last_day_df = disease_data.loc[disease_data["day"] == day]
                    infected = last_day_df.loc[last_day_df["state"] == "I"]["pid"].to_numpy()
                    recovered = last_day_df.loc[last_day_df["state"] == "R"]["pid"].to_numpy()
                    print(day+56, len(infected), len(recovered))
                    print('contacts', np.min(contact_matrix2), np.max(contact_matrix2), np.mean(contact_matrix2))
                    print('probs', np.min(probs), np.max(probs), np.mean(probs))
                    candidates = np.argpartition(probs, -18)[-18:].flatten().tolist()
                    top_inf = dict()
                    for cand in candidates:
                        top_inf[idx_to_pid[cand]] = probs[cand]
                    print(top_inf)

            if day >= 50:
                # Update contacts
                pid = pid_to_idx[row['pid1']]
                cid = pid_to_idx[row['pid2']]
                duration = row['duration'] / (60 * 24)  # Duration is in min, normalize to contact level over a day
                contact_increase = 1 - ((1 - transmission_rate) ** duration)
                inc2 = min(contact_matrix2[pid][cid] + contact_increase, 1)
                inc3 = min(contact_matrix3[cid][pid] + contact_increase, 1)
                contact_matrix2[pid][cid] = inc2
                contact_matrix2[cid][pid] = inc2
                contact_matrix3[pid][cid] = inc3
                contact_matrix3[cid][pid] = inc3
