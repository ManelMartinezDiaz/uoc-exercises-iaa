#proportion_factors.py
#support file for simulator calculations


#Proportion factors for each pair of imagery type

pf_RGB_term = (1, 1.2, 1.4, 1.7)
pf_RGB_IR = (1, 1.02, 1.05, 1.16)
pf_RGB_RADAR = (1.21, 1.7, 2.4, 2.5)
pf_RGB_LiDAR = (1.05, 2, 2.2, 2.25)
pf_RGB = {
    'term'  : pf_RGB_term,
    'IR'    : pf_RGB_IR,
    'RADAR' : pf_RGB_RADAR,
    'LiDAR' : pf_RGB_LiDAR
}

pf_term_RGB = (0.5, 1, 1, 1)
pf_term_IR = (2, 2.7, 3.1, 3.15)
pf_term_RADAR = (1.55, 1.6, 1.6, 1.6)
pf_term_LiDAR = (3.15, 4, 4.5, 4.75)
pf_term = {
    'RGB': pf_term_RGB,
    'IR': pf_term_IR,
    'RADAR': pf_term_RADAR,
    'LiDAR': pf_term_LiDAR
}

pf_IR_term = (1, 1.1, 1.14, 1.19)
pf_IR_RGB = (1, 1.1, 1.14, 1.19)
pf_IR_RADAR = (1, 1, 1.1, 1.1)
pf_IR_LiDAR = (1, 1, 1.8, 1.8)
pf_IR = {
    'term': pf_IR_term,
    'RGB': pf_IR_RGB,
    'RADAR': pf_IR_RADAR,
    'LiDAR': pf_IR_LiDAR
}

pf_RADAR_term = (2, 2.2, 2.4, 2.7)
pf_RADAR_IR = (1.7, 1.72, 1.77, 1.79)
pf_RADAR_RGB = (1, 1.13, 1.25, 1.46)
pf_RADAR_LiDAR = (1.4, 1.78, 2.15, 2.2)
pf_RADAR = {
    'term': pf_RADAR_term,
    'IR': pf_RADAR_IR,
    'RGB': pf_RADAR_RGB,
    'LiDAR': pf_RADAR_LiDAR
}

pf_LiDAR_term = (2, 2.2, 2.4, 2.7)
pf_LiDAR_IR = (2.1, 2.1, 2.1, 2.26)
pf_LiDAR_RADAR = (1.33, 1.66, 2, 2.5)
pf_LiDAR_RGB = (1.5, 1.65, 1.95, 1.95)
pf_LiDAR = {
    'term': pf_LiDAR_term,
    'IR': pf_LiDAR_IR,
    'RADAR': pf_LiDAR_RADAR,
    'RGB': pf_LiDAR_RGB
}

proportion_factors = {
    'RGB'   : pf_RGB,
    'term'   : pf_term,
    'IR'   : pf_IR,
    'RADAR'   : pf_RADAR,
    'LiDAR'   : pf_LiDAR,
}

def proportion_range_index(proportion):
    if proportion > 16:
        return 3
    elif proportion > 8:
        return 2
    elif proportion > 4:
        return 1
    elif proportion > 2:
        return 0
    else:
        return -1

def get_proportions(samples):
    return {
        'RGB': {
            'term'  : samples[0] / max(0.1, samples[1]), #to avoid division by zero
            'IR'    : samples[0] / max(0.1, samples[2]),
            'RADAR' : samples[0] / max(0.1, samples[3]),
            'LiDAR' : samples[0] / max(0.1, samples[4])
        },
        'term': {
            'RGB'   : samples[1] / max(0.1, samples[0]),
            'IR'    : samples[1] / max(0.1, samples[2]),
            'RADAR' : samples[1] / max(0.1, samples[3]),
            'LiDAR' : samples[1] / max(0.1, samples[4])
        },
        'IR': {
            'RGB'   : samples[2] / max(0.1, samples[0]),
            'term'  : samples[2] / max(0.1, samples[1]),
            'RADAR' : samples[2] / max(0.1, samples[3]),
            'LiDAR' : samples[2] / max(0.1, samples[4])
        },
        'RADAR': {
            'RGB'   : samples[3] / max(0.1, samples[0]),
            'term'  : samples[3] / max(0.1, samples[1]),
            'IR'    : samples[3] / max(0.1, samples[2]),
            'LiDAR' : samples[3] / max(0.1, samples[4])
        },
        'LiDAR': {
            'RGB'   : samples[4] / max(0.1, samples[0]),
            'term'  : samples[4] / max(0.1, samples[1]),
            'IR'    : samples[4] / max(0.1, samples[2]),
            'RADAR' : samples[4] / max(0.1, samples[3])
        }
    }