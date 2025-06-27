from utils import deg2num, num2deg

zoom_level = 20

# Seattle
## Roosevelt
# topleft_deg = (47.681744,-122.325996)  # topleft num: 167987, 365881
                                         # topleft deg: 47.681744, -122.325996
                                         # botright num: 168013.216442953, 365907.216442953
                                         # botright deg: 47.67550974029736, -122.31689651540462

## Northgate
# topleft_deg = (47.710431, -122.328963) # topleft num: 167979, 365757
                                         # topleft deg: 47.710431, -122.328963
                                         # botright num: 168005.216442953, 365783.216442953
                                         # botright deg: 47.704166826506054, -122.31964309743587


# LA
## Downtown Chinatown
# topleft_deg = (34.0631990, -118.2416627) #  Topleft_num:  (179884, 418650)
                                         #  Topleft_deg:  (34.063199, -118.2416627)
                                         #  Bottomright_num:  (179907, 418673)
                                         #  Bottomright_deg:  (34.056641640355934, -118.23348999023438)

## Altadena
# topleft_deg = (34.191647, -118.133806)  # Topleft_num:  (180198, 418198)
                                        # Topleft_deg:  (34.191647, -118.133806)
                                        # Bottomright_num:  (180221, 418221)
                                        # Bottomright_deg:  (34.18510984477344, -118.12568664550781)

## Ktown
# topleft_deg = (34.064376,-118.305435)   # Topleft_num:  (179698, 418646)
#                                         Topleft_deg:  (34.064376, -118.305435)
#                                         Bottomright_num:  (179721, 418669)
#                                         Bottomright_deg:  (34.057779382846725, -118.29734802246094)

## Torrance shopping center
# topleft_deg = (33.835593,-118.356391)   # Topleft_num:  (179550, 419450)
                                        # Topleft_deg:  (33.835593, -118.356391)
                                        # Bottomright_num:  (179573, 419473)
                                        # Bottomright_deg:  (33.828786509874305, -118.34815979003906)

## Glendale
# topleft_deg = (34.152738,-118.260768)   # Topleft_num:  (179828, 418335)
                                        # Topleft_deg:  (34.152738, -118.260768)
                                        # Bottomright_num:  (179851, 418358)
                                        # Bottomright_deg:  (34.14619208917146, -118.25271606445312)


# Massachusetts
## Acton stroad
# topleft_deg = (42.489280, -71.419658)   # Topleft_num:  (316263, 387324)
#                                         # Topleft_deg:  (42.48928, -71.419658)
#                                         # Bottomright_num:  (316286, 387347)
#                                         # Bottomright_deg:  (42.48323834594139, -71.4114761352539)

## Waltham shopping center
# topleft_deg = (42.372182, -71.222412)   # Topleft_num:  (316837, 387786)
                                        # Topleft_deg:  (42.372182, -71.222412)
                                        # Bottomright_num:  (316860, 387809)
                                        # Bottomright_deg:  (42.36615433532088, -71.21440887451172)


# NJ
# topleft_deg = (40.8036030,-74.4844324)  # Topleft_num:  (307336, 393895)
#                                         # Topleft_deg:  (40.803603, -74.4844324)
#                                         # Bottomright_num:  (307359, 393918)
#                                         # Bottomright_deg:  (40.797437319357236, -74.476318359375)


# DC
## USPS
topleft_deg = (38.921611, -76.995726)   # Topleft_num:  (300021, 401037)
                                        # Topleft_deg:  (38.921611, -76.995726)
                                        # Bottomright_num:  (300044, 401060)
                                        # Bottomright_deg:  (38.91534589480659, -76.98772430419922)

## NAtional mall union station
topleft_deg = (38.900216,-77.012636)# Topleft_num:  (299972, 401117)
                                    # Topleft_deg:  (38.900216, -77.012636)
                                    # Bottomright_num:  (299995, 401140)
                                    # Bottomright_deg:  (38.89397221118196, -77.00454711914062)

## Audi Field
topleft_deg = (38.8720477, -77.0170474) # Topleft_num:  (299959, 401223)
                                        # Topleft_deg:  (38.8720477, -77.0170474)
                                        # Bottomright_num:  (299982, 401246)
                                        # Bottomright_deg:  (38.86564216992963, -77.0090103149414)

onekm_in_num_tiles = 26.216442953
onekm_in_num_tiles = 24 # 915m
onekm_in_num_tiles = 23 # 915m


topleft_num = deg2num(topleft_deg[0], topleft_deg[1], zoom_level)
bottomright_num = (topleft_num[0] + onekm_in_num_tiles, topleft_num[1] + onekm_in_num_tiles)

bottomright_deg = num2deg(bottomright_num[0]+1, bottomright_num[1]+1, zoom_level)

print("Topleft_num: ", topleft_num)
print("Topleft_deg: ", topleft_deg)
print("Bottomright_num: ", bottomright_num)
print("Bottomright_deg: ", bottomright_deg)
