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

    si_table.set_index(["pid", "infected"], verify_integrity=True, inplace=True)

    return si_table



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("graph_file", help="edges csv object")
    parser.add_argument("disease_file", help="disease data")
    parser.add_argument("pop_file", help="population characteristics file")
    # parser.add_argument("out_file", help="output file for data")
    parser.add_argument("--min-date", default=0, type=int, help="Min date to generate data from (used for making evaluation set)")
    parser.add_argument("--is-eval", action="store_true",default=False, help="exclude positive instances (assume input file is training)") 
    parser.add_argument("--past-window", default=3, help="How much history to consider for each data point")
    # parser.add_argument("--pid_partition", type=int, help="index of pids partition")
    # parser.add_argument("--n_jobs", type=int, help="total number of jobs")

    args = parser.parse_args()

    pop = pd.read_csv(args.pop_file)
    disease_data = pd.read_csv(args.disease_file)
    pop.set_index("pid", inplace=True)


    #######################################  Start collecting data fields  #################################
    # TODO: I'm not sure how to obtain the transmission rate - how is it defined? What does the location mean?
    # Nonetheless, these data should probably be extracted from the following contact graph
    graph = make_graph(args.graph_file)

    # This is a table which stores all the infection information
    # At the moment it is not used, but it could be useful
    si_table = make_si_table(disease_data)
    disease_unused = si_table[si_table.index.get_level_values("infected") >= (args.min_date - args.past_window)]

    # Process the training data and get the following info related to the last training day: 
    num_of_training_days = 56
    last_day_df = disease_data.loc[disease_data["day"] == num_of_training_days]

    # Total population size
    total_population_size = len(last_day_df)

    # Size of population infected at the end of the training period
    # Question: should this include "recovered"? Right now it does
    # Otherwise, switch to: len(last_day_df.loc[last_day_df["state"] == "I"])
    size_of_infected_at_last_day = len(last_day_df.loc[last_day_df["state"] != "S"]) 

    # Person IDs which have not yet been infected at the end of the training period
    pid_not_infected_at_last_day = last_day_df.loc[last_day_df["state"] == "S"]["pid"].to_numpy()

    # Contact between those IDs for the testing period
    # Question: do we not care about their contact during training period? Right now we retrieve all contact information
    # Also, fee free to process the subgraph based on wh
    subgraph_list = []
    for ego_pid in pid_not_infected_at_last_day:
        try:
            vid = graph.vs.find(name=ego_pid)
        except:
            # Some node doesn't have contact
            print(f"pid {ego_pid} has no contact history")
            continue
        neighbors = graph.neighborhood(vid, order=1) # changing the order to get more contacts
        subgraph = graph.induced_subgraph(neighbors+[vid])
        subgraph_list.append(subgraph)

        # We can also get information about each edge
        # Right now only includes duration. We can include more information by modifying the make_graph file
        for edge in subgraph.es:
            edge_info = edge["duration"]


    print("From here, we can either store the processed data / directly train model based on the processed data")
    import ipdb
    ipdb.set_trace()
    