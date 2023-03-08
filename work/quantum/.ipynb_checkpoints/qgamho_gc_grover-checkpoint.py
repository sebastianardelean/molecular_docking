
import warnings; warnings.simplefilter('ignore')

from qiskit import QuantumCircuit, execute, Aer, IBMQ,QuantumRegister,ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import *
import networkx as nx

import operator


import math
import random
from qiskit import transpile, schedule as build_schedule
from qiskit.test.mock import FakeAlmaden

import pygad
import numpy as np

import csv

from qiskit import IBMQ
IBMQ.save_account("2a8c9616aea0e7a684e9b4c3e2efc756847f326adbb130b86a3ccb0b6fca805fa4c2a45ba6fbea9624888efb55bc9ddfccc71add0e7599b20be354ef82f5642b")

"""# Genetic Algorithm"""

METAHEURISTIC_GENE_LENGTH:int = 5
METAHEURISTIC_GENE_LOW:int = 10 
METAHEURISTIC_GENE_HIGH:int = 54 #unit digit = color, decimal digit => node; example: 23->node 2 has color 3
METAHEURISTIC_NUM_PARENS_MATING:int=2
METAHEURISTIC_POPULATION_SIZE:int = 1000
METAHEURISTIC_SELECTION_TYPE:str="rws"
METAHEURISTIC_KEEP_PARENTS:int=1
METAHEURISTIC_CROSSOVER_RATE:float = 0.6
METAHEURISTIC_CROSSOVER_TYPE:str = "single_point"
METAHEURISTIC_MUTATION_TYPE:str = "random"
METAHEURISTIC_MUTATION_RATE: float = 0.00002
METAHEURISTIC_NO_OF_GENERATIONS:int = 500
COLOR_LIST = [(0,0),(0,1),(1,0),(1,1)]
INVALID_COLOR_CODE = 3

NO_OF_NODES = 5
NO_OF_COLORS = 3# 2 bits needed
INVALID_COLORS_LIST = [(1,1)]
NO_OF_QUBITS_FITNESS = 8
NO_OF_QUBITS_PER_COLOR = 2
NO_OF_QUBITS_INDIVIDUAL = NO_OF_QUBITS_PER_COLOR*NO_OF_NODES
POPULATION_SIZE = 2**NO_OF_QUBITS_INDIVIDUAL #2*NO_OF_NODES#
NO_OF_MAX_GROVER_ITERATIONS = int(math.sqrt(2**(NO_OF_QUBITS_INDIVIDUAL - 2*METAHEURISTIC_GENE_LENGTH)))-1

def to_binary(value, number_of_bits, lsb=False):
    """
    Function return two's complement representation
    :param value: value in decimal representation
    :param number_of_bits: number of bits used for representation
    :returns: np.array that represents the binary representation
    >>> to_binary(10,4)
    array([1, 0, 1, 0])
    >>> to_binary(10,4,True)
    array([0, 1, 0, 1])
    """
    if lsb == True:
        return flip(array(list(binary_repr(value, number_of_bits)), dtype=int))
    return array(list(binary_repr(value, number_of_bits)), dtype=int)

def pairs_colors(colors_list):
    """
    Function returns a list of colors from the binary representation of the individual
    :param colors_list: binary representation of the individual
    :returns: list of pairs representing the color configuration for each node.
    """
    pairs=list()
    for i in range(0,len(colors_list),2):
        pairs.append((colors_list[i],colors_list[i+1]))
    return pairs

def check_edges_validity(graph, colors):
    """
    Function return the number of edges between adjacent nodes colored using different colors
    :param graph: adjacency matrix
    :param colors: list of colors
    :returns: number of edges between adjacent nodes colored using different colors
    """
    no_of_valid_edges = 0
    for color in colors:
        if color in INVALID_COLORS_LIST:
            return -1
    for i in range(0,NO_OF_NODES):
        for j in range(i + 1, NO_OF_NODES):
            if graph[i][j]:#daca am legatura
                if colors[j]==colors[i]:
                    continue
                else:
                    no_of_valid_edges +=1

    return no_of_valid_edges

def get_number_of_edges(graph):
    """
    Function return the number of edges in graph
    :param graph: adjacency matrix
    :returns: number of edges in graph
    """
    no_of_edges = 0
    for i in range(NO_OF_NODES):
        for j in range(i + 1, NO_OF_NODES):
            if graph[i][j]:
                no_of_edges +=1    
    return no_of_edges

GRAPH = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]]

def get_best_solution(adjacency_matrix):
    nx_graph = nx.from_numpy_matrix(np.array([np.array(xi) for xi in adjacency_matrix]))
    colors = nx.greedy_color(nx_graph,strategy='largest_first', interchange = False)
    solution = []
    for node, color in colors.items():
        solution.append(color)
    return solution

print(get_best_solution(GRAPH))

graph = nx.from_numpy_matrix(np.array([np.array(xi) for xi in GRAPH]))
nx.draw_networkx(graph, with_labels=True)

def check_duplicates(input_list):
    seen = set()
    uniq = []
    for el in input_list:
        if el not in seen:
            uniq.append(el)
            seen.add(abs(el))
    return uniq

def calculate_fitness(graph, nodes, colors):
    """
    Function return the number of edges between adjacent nodes colored using different colors
    :param graph: adjacency matrix
    :param colors: list of colors
    :returns: number of edges between adjacent nodes colored using different colors
    """
    no_of_valid_edges = 0
    for color in colors:
        if color == INVALID_COLOR_CODE:
            return -1
    for i in range(0,len(nodes)):
        for j in range(i+1,len(nodes)):
            if graph[i][j]:
                if colors[j]==colors[i]:
                    continue
                else:
                    no_of_valid_edges +=1

    return no_of_valid_edges

#example: 23->node 2 has color 3
#Accepts a function that must accept 2 parameters (a single solution and its index in the population) and return the fitness value of the solution
def fitness_func(solution, solution_idx):
    node_list = solution // 10
    node_list = node_list - 1
    color_list = solution % 10
    #check if nodes are unique
    unique_elements = check_duplicates(node_list)
    if len(unique_elements) < len(node_list):
        return -2
    #check if colors are in range
    for item in color_list:
        if item > len(COLOR_LIST)-1:
            return -3
    #check if nodes are in range
    for node in node_list:
        if node>NO_OF_NODES:
            return -4
    fitness = calculate_fitness(GRAPH,node_list,color_list);
    return fitness

def run_ga():
  ga_instance = pygad.GA(num_generations=METAHEURISTIC_NO_OF_GENERATIONS,
                       num_parents_mating=METAHEURISTIC_NUM_PARENS_MATING,
                       gene_type = int,
                       fitness_func=fitness_func,
                       sol_per_pop=METAHEURISTIC_POPULATION_SIZE,
                       num_genes=METAHEURISTIC_GENE_LENGTH,
                       init_range_low=METAHEURISTIC_GENE_LOW,
                       init_range_high=METAHEURISTIC_GENE_HIGH,
                       parent_selection_type=METAHEURISTIC_SELECTION_TYPE,
                       keep_parents=METAHEURISTIC_KEEP_PARENTS,
                       crossover_type=METAHEURISTIC_CROSSOVER_TYPE,
                       crossover_probability = METAHEURISTIC_CROSSOVER_RATE,
                       mutation_type=METAHEURISTIC_MUTATION_TYPE,
                       mutation_probability =METAHEURISTIC_MUTATION_RATE,
                       stop_criteria="saturate_10")
  ga_instance.run()
  solution, solution_fitness, solution_idx = ga_instance.best_solution()
  print("Parameters of the best solution : {solution}".format(solution=solution))
  print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
  print("Number of generations={generations}".format(generations=ga_instance.best_solution_generation))
  return solution.tolist(), ga_instance.generations_completed

"""# RQGA"""

def to_binary(value, number_of_bits, lsb=False):
    """
    Function return two's complement representation
    :param value: value in decimal representation
    :param number_of_bits: number of bits used for representation
    :returns: np.array that represents the binary representation
    >>> to_binary(10,4)
    array([1, 0, 1, 0])
    >>> to_binary(10,4,True)
    array([0, 1, 0, 1])
    """
    if lsb == True:
        return np.flip(np.array(list(np.binary_repr(value, number_of_bits)), dtype=int))
    return np.array(list(np.binary_repr(value, number_of_bits)), dtype=int)

def get_marked_individuals(individual_mho, individual_binary):
    marked_individual=[]
    is_marked = False
    if METAHEURISTIC_GENE_LENGTH == 0:
        for value in individual_binary:
            marked_individual.append((value[0],value[1],False))
        is_marked=True
    else:
        for value in individual_binary:
            marked_individual.append((value[0],value[1],False))
        for gene in individual_mho:
            color = gene % 10 
            node = gene // 10
            node = node - 1
            color = COLOR_LIST[color]
            # if individual_binary[node] == color:
            #     value = marked_individual[node]
            #     new_value = (value[0],value[1],True)
            #     marked_individual[node]=new_value
            value = marked_individual[node]
            new_value = (color[0],color[1],True)
            marked_individual[node]=new_value

        marked_only_size = len(list(filter(lambda x:(x[1]==True), marked_individual)))
        if marked_only_size == len(individual_mho):
            is_marked = True
    return marked_individual,is_marked

def string_to_list(string):
    return_list = []
    for i in string:
        if i != " ":
            return_list.append(int(i))
    return return_list

def recreate_individual(individual_mho, individual_binary):
    colors = pairs_colors(individual_binary)
    solution = [(0,0,False) for _ in range(0,NO_OF_NODES)]
    if METAHEURISTIC_GENE_LENGTH == 0:
        return colors
    for gene in individual_mho:
        color = gene %10
        color = COLOR_LIST[color]
        node = gene // 10
        node = node - 1
        solution[node]=(color[0],color[1],True)

    for gene in colors:
        for i in range(0,len(solution)):
            value = solution[i]
            if value[2]==False:
                new_value = (gene[0],gene[1],True)
                solution[i]=new_value
                break
    return list(map(lambda x: (x[0],x[1]), solution))

def get_ufit_instruction(individual_mho):
    #define and initialize the individual quantum register
    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL-2*METAHEURISTIC_GENE_LENGTH,"ind_qreg")
    #define and initialize the fitness quantum register. 
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fit_qreg")
    #create the ufit subcircuit
    qc = QuantumCircuit(ind_qreg,fit_qreg,name="U$_fit$")
    for i in range(0,POPULATION_SIZE):
        """
        For each individual in population get the two's complement representation and 
        set the qubits on 1 using X-gate, according to the binary representation
        """
        individual_binary = to_binary(i, NO_OF_QUBITS_INDIVIDUAL, True)

        colors = pairs_colors(individual_binary)

        marked_individual,is_marked=get_marked_individuals(individual_mho,colors)
       # print("Size of marked_individual:{0}\t {1}", len(marked_individual), marked_individual)
        if not is_marked:
            continue            
        not_marked_individuals=[]
        for value in marked_individual:
            if value[2]==False:
                not_marked_individuals.append(value[0])
                not_marked_individuals.append(value[1])
       # print("Size of not_marked_individuals:{1}",len(not_marked_individuals))
        for qubit in range(0,len(not_marked_individuals)):
            if not_marked_individuals[qubit]==0:
                qc.x(ind_qreg[qubit])



        """
        Calculate the fitness value and get the two's complement representation of the fitness value.
        """    
        new_individual_binary = list(map(lambda x : (x[0],x[1]), marked_individual))
        #calculate valid score
        valid_score = check_edges_validity(GRAPH,new_individual_binary)
        valid_score_binary = to_binary(valid_score,NO_OF_QUBITS_FITNESS,True)
        

        """
        Set the fitness value in fitness quantum register for each individual and mark it valid or invalid
        """
        for k in range(0,NO_OF_QUBITS_FITNESS):
            if valid_score_binary[k]==1:
                qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL-2*METAHEURISTIC_GENE_LENGTH)],fit_qreg[k])

        #if fitness value si greater than 0 then set the valid qubit to 1
        if valid_score > 0:
            qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL-2*METAHEURISTIC_GENE_LENGTH)],fit_qreg[NO_OF_QUBITS_FITNESS])
        #reset individual

        for qubit in range(0,len(not_marked_individuals)):
            if not_marked_individuals[qubit]==0:
                qc.x(ind_qreg[qubit])
        qc.barrier()
    return qc.to_instruction()

def get_oracle_instruction(positive_value_array):
    #define and initialize fitness quantum register
    fit_reg = QuantumRegister(NO_OF_QUBITS_FITNESS,"fqreg")
    #define and initialize max quantum register
    no_of_edges_reg=QuantumRegister(NO_OF_QUBITS_FITNESS,"noqreg")
    #define and initialize carry quantum register
    carry_reg = QuantumRegister(3,"cqreg")
    #define and initialize oracle workspace quantum register
    oracle = QuantumRegister(1,"oqreg")
    #create Oracle subcircuit
    oracle_circ = QuantumCircuit(fit_reg,no_of_edges_reg,carry_reg,oracle,name="O")
    
    #define majority operator
    def majority(circ,a,b,c):
        circ.cx(c,b)
        circ.cx(c,a)
        circ.ccx(a, b, c)
    #define unmajority operator
    def unmaj(circ,a,b,c):
        circ.ccx(a, b, c)
        circ.cx(c, a)
        circ.cx(a, b)
    #define the Quantum Ripple Carry Adder
    def adder_8_qubits(p,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,cin,cout):
        majority(p, cin, b0, a0)
        majority(p, a0, b1, a1)
        majority(p, a1, b2, a2)
        majority(p, a2, b3, a3)
        majority(p, a3, b4, a4)
        majority(p, a4, b5, a5)
        majority(p, a5, b6, a6)
        majority(p, a6, b7, a7)
        
        p.cx(a7, cout)
        
        unmaj(p, a6, b7, a7)
        unmaj(p, a5, b6, a6)
        unmaj(p, a4, b5, a5)
        unmaj(p, a3, b4, a4)
        unmaj(p, a2, b3, a3)
        unmaj(p, a1, b2, a2)
        unmaj(p, a0, b1, a1)
        unmaj(p, cin, b0, a0)
    
    """
    Subtract max value. We start by storing the max value in the quantum register. Such, considering that 
    all qubits are |0>, if on position i in positive_value_array there's 0, then qubit i will be negated. Otherwise, 
    if on position i in positive_value_array there's a 1, by default will remain 0 in no_of_edges_reg quantum
    register. For performing subtraction, carry-in will be set to 1.
    """
    for i in range(0,NO_OF_QUBITS_FITNESS):
        if positive_value_array[i]==0:
            oracle_circ.x(no_of_edges_reg[i])
    oracle_circ.x(carry_reg[0])

    adder_8_qubits(oracle_circ, 
            no_of_edges_reg[0],no_of_edges_reg[1],no_of_edges_reg[2],no_of_edges_reg[3],
            no_of_edges_reg[4],no_of_edges_reg[5],no_of_edges_reg[6],no_of_edges_reg[7],       
            fit_reg[0],fit_reg[1],fit_reg[2],fit_reg[3],
            fit_reg[4],fit_reg[5],fit_reg[6],fit_reg[7],
               carry_reg[0],carry_reg[1]);

    
    oracle_circ.barrier()
    """
    Reset the value in no_of_edges_reg and carry-in
    """
    oracle_circ.x(no_of_edges_reg)
    oracle_circ.x(carry_reg[0])
    
    """
    Mark the corresponding basis states by shifting their amplitudes.
    """
    
    oracle_circ.h(oracle[0])
    oracle_circ.mct([fit_reg[i] for i in range(0,NO_OF_QUBITS_FITNESS)],oracle[0])
    oracle_circ.h(oracle[0])
    
    """
    Restore the fitness value by adding max value.
    """
    adder_8_qubits(oracle_circ, 
            no_of_edges_reg[0],no_of_edges_reg[1],no_of_edges_reg[2],no_of_edges_reg[3],
            no_of_edges_reg[4],no_of_edges_reg[5],no_of_edges_reg[6],no_of_edges_reg[7],       
            fit_reg[0],fit_reg[1],fit_reg[2],fit_reg[3],
            fit_reg[4],fit_reg[5],fit_reg[6],fit_reg[7],
            carry_reg[0],carry_reg[2]);
    return oracle_circ.to_instruction()

def get_grover_iteration_subcircuit():
    #define and initialize fitness quantum register
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fqreg")
    #define and initialize oracle workspace quantum register
    oracle_ws = QuantumRegister(1,"ows")
    #create grover diffuser subcircuit
    grover_circ = QuantumCircuit(fit_qreg,oracle_ws,name ="U$_s$")

    grover_circ.h(fit_qreg)
    grover_circ.x(fit_qreg)

    grover_circ.h(oracle_ws[0])

    grover_circ.mct(list(range(NO_OF_QUBITS_FITNESS+1)), oracle_ws[0])  # multi-controlled-toffoli

    grover_circ.h(oracle_ws[0])


    grover_circ.x(fit_qreg)
    grover_circ.h(fit_qreg)
    grover_circ.h(oracle_ws)

    return grover_circ.to_instruction()

def run_algorithm(run_no, writer):
    #Load IBMQ account
    IBMQ.load_account()
    #calculate the number of edges in graph
    pos_no_of_edges = get_number_of_edges(GRAPH)
    print("No of edges:{0}".format(pos_no_of_edges))
    #define a list for storing the results
    
    

    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL-2*METAHEURISTIC_GENE_LENGTH,"ireg")
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"freg") #8 qubits fitness + 1 valid
    carry_qreg = QuantumRegister(2*NO_OF_MAX_GROVER_ITERATIONS+1,"qcarry")
    oracle = QuantumRegister(1,"oracle")
    creg = ClassicalRegister(NO_OF_QUBITS_INDIVIDUAL-2*METAHEURISTIC_GENE_LENGTH,"reg")
    no_of_edges_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS,"pos_max_qreg")
    
    print("Creating quantum circuit...")
    number_of_ga_generations = 0
    qc = QuantumCircuit(ind_qreg,fit_qreg,carry_qreg,oracle,no_of_edges_qreg,creg)

    
    print("Metaheuristic...")
    if METAHEURISTIC_GENE_LENGTH == NO_OF_NODES:
        individual_mho, number_of_ga_generations = run_ga()
        writer.writerow({"algo_run_no":run_no,"solution":individual_mho, "ga_generation":number_of_ga_generations})
        return;      
    elif METAHEURISTIC_GENE_LENGTH == 0:
        #pure quantum, no need to call run_ga
        individual_mho=[]
    else:
        individual_mho,number_of_ga_generations = run_ga()
    print("Creating superposition of individuals...")
    qc.h(ind_qreg)
    qc.h(oracle)

    print("Getting maximum number of edges {0} binary representation...".format(pos_no_of_edges))
    pos_value_bin = to_binary(pos_no_of_edges,NO_OF_QUBITS_FITNESS,True)


    print("Getting ufit, oracle and grover iterations subcircuits...")
    ufit_instr = get_ufit_instruction(individual_mho)
    oracle_instr = get_oracle_instruction(pos_value_bin)
    grover_iter_inst = get_grover_iteration_subcircuit()

    print("Append Ufit instruction to circuit...")
    qc.append(ufit_instr, [ind_qreg[q] for q in range(0,NO_OF_QUBITS_INDIVIDUAL-2*METAHEURISTIC_GENE_LENGTH)]+
                            [fit_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS+1)]
            )
        
    for it in range(0,NO_OF_MAX_GROVER_ITERATIONS):
        print("Append Oracle instruction to circuit...")
            
        qc.append(oracle_instr,[fit_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS)]+
                            [no_of_edges_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS)]+
                            [carry_qreg[0],carry_qreg[2*it+1],carry_qreg[2*it+2]]+
                            [oracle[0]])
        print("Append Grover Diffuser to circuit...")
        qc.append(grover_iter_inst, [fit_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS+1)]+[oracle[0]])
        
    print("Measure circuit...")
    qc.measure(ind_qreg,creg)

    
    simulation_results = []
    
    provider = IBMQ.get_provider(hub='ibm-q',group='open', project='main')
    backend = provider.get_backend('simulator_mps')

    # Perform 10 measurements for each circuit
    
    print("Setup simulator...")    
    shots = 16
    try:
        print("Starting simulator...")
        mapped_circuit = transpile(qc, backend=backend)
        qobj = assemble(mapped_circuit, backend=backend, shots=shots)
        runner = backend.run(qobj)
        job_monitor(runner)
        results = runner.result()
        answer = results.get_counts()
        #Get the result with the maximum number of counts
        max_item =max(answer.items(), key=operator.itemgetter(1))
        solution_individual = string_to_list(max_item[0])
        solution_individual = recreate_individual(individual_mho,solution_individual)
        new_fitness_value=check_edges_validity(GRAPH,solution_individual)
        #Store the result.
        writer.writerow({"algo_run_no":run_no,"solution":solution_individual, "fitness_value":new_fitness_value,"rqga_generation":1,"ga_generation":number_of_ga_generations})
        print("Found solution {0} with fitness {1}...".format(solution_individual,new_fitness_value))
    except Exception as e:
        print(str(e))

from tqdm import tqdm
import sys

import contextlib

with open("results.csv", "w", newline='') as csvfile:
    fieldnames = ["algo_run_no", "solution", "fitness_value","rqga_generation", "ga_generation"]
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()

    for i in tqdm(range(100)):
        run_algorithm(i,writer)

