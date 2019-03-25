#analyzer.py
#analyze and simulator functions definitios

import random
import numpy as np
import proportion_factors as pf

####################
####Exercise 2b#####
####################
def decode(individual):
    decoded_individual = (None,None,None,None,None)

    #TO DO: Decode the received individual so that this function returns a tuple of 5 integers
    #type = 0;
    #numberImages = 0
    
    #print("LEN:", len(individual))
    #for item in individual:
        #decoded_individual[item] = individual[item]
        #type += items[item][0]
    #    numberImages = individual[item]
    #    print("Decode_List:",numberImages)
    #    decoded_individual.append(ITEMS[item][1])

    decoded_individual = (individual[1],
                          individual[3],
                          individual[5],
                          individual[7],
                          individual[9]
                          )
        
    return decoded_individual


def encode(decoded_individual):
    
    encoded_individual = (1,decoded_individual[0],
                          2,decoded_individual[1],
                          3,decoded_individual[2],
                          4,decoded_individual[3],
                          5,decoded_individual[4],
                          )
    
    return encoded_individual

#trivial decode function: return a tuple with random numbers between 50 and 100
def decode_trivial(individual):

    decoded_individual = (
        random.randint(50, 100),
        random.randint(50, 100),
        random.randint(50, 100),
        random.randint(50, 100),
        random.randint(50, 100)
    )

    return decoded_individual


def simulator(samples, needed_samples=(2000,500,1000,300,70), include_bias=False):

    if type(samples) is not tuple:
        raise TypeError('Argument "samples" must be a tuple!')

    elif len(samples) != 5:
        raise Exception('Argument "samples" must have length 5!')

    try:
        total_ops = sum(samples)
        if total_ops > 1500:
            raise Exception('Too much samples to process! Maximum: 1000')

    except TypeError:
        print('All values in "samples" must be integers')

    tbase = (101.000, 355.800, 420.000, 726.000, 890.000)

    if include_bias:
        times = list(map(lambda x: x*random.uniform(0.985,1.015),np.array(samples) * np.array(tbase)))
    else:
        times = list(np.array(samples) * np.array(tbase))

    proportions = pf.get_proportions(samples)

    image_type_index = 0

    for image_type in ('RGB','term','IR','RADAR','LiDAR'):

        for image_type_2 in tuple(filter(lambda x: x != image_type,('RGB','term','IR','RADAR','LiDAR'))):

            prop_range_index = pf.proportion_range_index(proportions[image_type][image_type_2])

            if prop_range_index >= 0:
                times[image_type_index] *= pf.proportion_factors[image_type][image_type_2][prop_range_index]

        image_type_index += 1

    remaining_RGB = max(0, min(2000, needed_samples[0]) - samples[0])
    remaining_term = max(0, min(500, needed_samples[1]) - samples[1])
    remaining_IR = max(0, min(1000, needed_samples[2]) - samples[2])
    remaining_RADAR = max(0, min(300, needed_samples[3]) - samples[3])
    remaining_LiDAR = max(0, min(70, needed_samples[4])- samples[4])
    
    remaining_samples = (remaining_RGB,remaining_term,remaining_IR,remaining_RADAR,remaining_LiDAR)

    return sum(times),remaining_samples


def analyze_performance(individual):
    ####################
    ####Exercise 2######
    ####################
    #TO DO: replace the function call of tnext line with a call to the function 'decode' that you just implemented
    #samples = decode_trivial(individual)
    samples = decode(individual)

    remaining_samples = (2000, 500, 1000, 300, 70)
    analysis_time = 0

    while sum(remaining_samples) > 0:
        #Exercise 4 - Comment next line:
        #sim_time,remaining_samples = simulator(samples,remaining_samples)
        #Exercise 4 - Uncomment next line:
        sim_time,remaining_samples = simulator(samples,remaining_samples,True)

        analysis_time += sim_time

    return analysis_time,