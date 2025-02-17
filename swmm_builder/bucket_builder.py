def sub_catchment_polygons(nodes, x_coord, y_coord, interval_x=0, interval_y=0):
    for node in nodes:
        print(f"c_{node}_catchment  	{x_coord}         	{y_coord}          ")
        y_coord += interval_y
        x_coord += interval_x


def node_coordinates(nodes, x_coord, y_coord, interval_x=0, interval_y=0):
    for node in nodes:
        print(f"c_{node}_vr_storage   	{x_coord}         	{y_coord}          ")
        y_coord += interval_y
        x_coord += interval_x


def subcatchments(nodes, areas, gage_name):
    for node, area in zip(nodes, areas):
        print(
            f"c_{node}_catchment 	rain_gage_{gage_name}	c_{node}_vr_storage	{area}       	50      	400     	0.5     	0       	                "
        )


def subareas(nodes, s_imperv):
    for node in nodes:
        print(
            f"c_{node}_catchment  	0.01      	0.1       	{s_imperv}         	0.05      	25        	OUTLET    "
        )


def infiltration(nodes):
    for node in nodes:
        print(
            f"c_{node}_catchment 	3.5       	0.5       	0.26      	          	          "
        )


def vr_storage(nodes):
    for node in nodes:
        print(
            f"c_{node}_vr_storage	0       	10        	0         	PYRAMIDAL 	1000      	1000      	0       	0        	0       "
        )


def vr_pumps(nodes, delays, target):
    for node, delay in zip(nodes, delays):
        print(
            f"c_{node}_d_{delay}       	c_{node}_vr_storage	c_{node}_vr_storage    	virtual_pump_curve	ON      	0       	0       "
        )


def dwf(nodes, baselines, target):
    for node, baseline in zip(nodes, baselines):
        print(
            f'c_{node}_vr_storage 	FLOW            	{baseline:.5f}     "hourly_{target}"	"daily_{target}"	"weekend_{target}"	"monthly_{target}"'
        )


# nodes_eindhoven = [24, 23, 22, 21, 12, 20, 19, 10, 16, 15, 14]

# nodes_geldrop = [129, 128, 1285, 127, 126, 125]
# areas_geldrop = [5, 255.1, 5.8, 134.9, 15.8, 5.3]
# delays_geldrop = [2, 64, 10, 46, 16, 8]
# baselines_geldrop = [
#     (5 * 0) * 0.19 / 24 / 3600 + 0,
#     (255.1 * 9059) * 0.19 / 24 / 3600 + 0.05848,
#     (5.8 * 8879) * 0.19 / 24 / 3600 + 0,
#     (134.9 * 6628) * 0.19 / 24 / 3600 + 0,
#     (15.8 * 6633) * 0.19 / 24 / 3600 + 0.02297,
#     (5.3 * 6566) * 0.19 / 24 / 3600 + 0,
# ]

# nodes_geldropW = [136, 135, 131, 130, 134, 132]
# areas_geldropW = [50.5, 22, 21.5, 3.9, 5.3, 3.9]
# delays_geldropW = [28, 18, 18, 8, 8, 8]
# baselines_geldropW = [
#     (50.5 * 12006) * 0.19 / 24 / 3600 + 0.016,
#     (22 * 15727) * 0.19 / 24 / 3600 + 0,
#     (21.5 * 15302) * 0.19 / 24 / 3600 + 0,
#     (3.9 * 25667) * 0.19 / 24 / 3600 + 0,
#     (5.3 * 13925) * 0.19 / 24 / 3600 + 0,
#     (3.9 * 18632) * 0.19 / 24 / 3600 + 0,
# ]

nodes_RZ = [
    150,
    123,
    143,
    147,
    160,
    145,
    144,
    142,
    148,
    149,
    151,
    161,
    138,
    139,
    137,
    122,
    121,
    120,
    119,
    100,
    118,
    116,
    114,
    112,
    107,
    '107_2',
    106,
    99,
    103,
    98,
]
areas_RZ = [
    205,
    101.5,
    308,
    4.3,
    5,
    1,
    22.8,
    76.8,
    11.2,
    18.3,
    29.4,
    68,
    28.9,
    4.6,
    1,
    52.2,
    10.3,
    4.9,
    393,
    6.7,
    11.3,
    12.9,
    6.1,
    28.5,
    77.2,
    9,
    11.2,
    103.7,
    3.8,
    26.2,
]
delays_RZ = [
    57,
    40,
    70,
    4,
    8,
    4,
    19,
    35,
    12,
    17,
    22,
    32,
    22,
    8,
    4,
    29,
    12,
    8,
    80,
    10,
    13,
    15,
    10,
    21,
    36,
    12,
    13,
    41,
    8,
    20,
]
baselines_RZ = [
    (205 * 3844) * 0.19 / 24 / 3600 + 0.01,
    (101.5 * 10390) * 0.19 / 24 / 3600 + 0,
    (308 * 30422) * 0.19 / 24 / 3600 + 0,
    (4.3 * 0) * 0.19 / 24 / 3600 + 0,
    (5 * 1143340) * 0.19 / 24 / 3600 + 0,
    (1 * 74800) * 0.19 / 24 / 3600 + 0,
    (22.8 * 8211) * 0.19 / 24 / 3600 + 0,
    (76.8 * 11837) * 0.19 / 24 / 3600 + 0,
    (11.2 * 8982) * 0.19 / 24 / 3600 + 0,
    (18.3 * 0) * 0.19 / 24 / 3600 + 0,
    (29.4 * 2595) * 0.19 / 24 / 3600 + 0,
    (68 * 0) * 0.19 / 24 / 3600 + 0,
    (14619 * 28.9) * 0.19 / 24 / 3600 + 0,
    (0) * 0.19 / 24 / 3600 + 0,
    (1 * 191700) * 0.19 / 24 / 3600 + 0,
    (52.5 * 10347) * 0.19 / 24 / 3600 + 0,
    (10.3 * 13029) * 0.19 / 24 / 3600 + 0.00678,
    (4.9 * 27143) * 0.19 / 24 / 3600 + 0,
    (393 * 8844) * 0.19 / 24 / 3600 + 0,
    (6.7 * 11000) * 0.19 / 24 / 3600 + 0,
    (11.3 * 0) * 0.19 / 24 / 3600 + 0,
    (12.9 * 11760) * 0.19 / 24 / 3600 + 0,
    (6.1 * 13131) * 0.19 / 24 / 3600 + 0,
    (28.5 * 7039) * 0.19 / 24 / 3600 + 0,
    (77.2 * 10099) * 0.19 / 24 / 3600 + 0.118,
    (9 * 100) * 0.19 / 24 / 3600 + 0.00027,
    (11.2 * 15518) * 0.19 / 24 / 3600 + 0.0104,
    (103.7 * 7756) * 0.19 / 24 / 3600 + 0,
    (3.8 * 18211) * 0.19 / 24 / 3600 + 0,
    (26.2 * 11748) * 0.19 / 24 / 3600 + 0,
]
