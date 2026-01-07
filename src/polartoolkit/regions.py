"""
Bounding regions for commonly plotted polar regions. In stereographic projections. The
format is (xmin, xmax, ymin, ymax), in meters.
"""

import typing

import pandas as pd
import verde as vd
from shapely import Polygon

from polartoolkit import (  # pylint: disable=import-self
    maps,
    regions,  # noqa: PLW0406
    utils,
)

try:
    import ipyleaflet
except ImportError:
    ipyleaflet = None


try:
    from IPython.display import display
except ImportError:
    display = None

#####
#####
# Antarctica
#####
#####

# regions
antarctica = (-2800e3, 2800e3, -2800e3, 2800e3)
west_antarctica = (-2740e3, 570e3, -2150e3, 1670e3)
east_antarctica = (-840e3, 2880e3, -2400e3, 2600e3)
antarctic_peninsula = (-2600e3, -1200e3, 170e3, 1800e3)
marie_byrd_land = (-1500e3, -500e3, -1350e3, -800e3)
victoria_land = (100e3, 1000e3, -2200e3, -1000e3)
# wilkes_land
# queen_maud_land
saunders_coast = (-980e3, -600e3, -1350e3, -1100e3)

# study_sites
roosevelt_island = (-480e3, -240e3, -1220e3, -980e3)
ross_island = (210e3, 360e3, -1400e3, -1250e3)
minna_bluff = (210e3, 390e3, -1310e3, -1120e3)
# discovery_deep
mcmurdo_dry_valleys = (320e3, 480e3, -1400e3, -1220e3)
siple_coast = (-700e3, 30e3, -1110e3, -450e3)
crary_ice_rise = (-330e3, -40e3, -830e3, -480e3)
siple_dome = (-630e3, -270e3, -970e3, -630e3)

# ice_shelves
# WEST ANTARCTICA
# manually created ice shelf regions
# created from code at bottom of this script
ross_ice_shelf = (-680e3, 470e3, -1420e3, -310e3)
nickerson_ice_shelf = (-980e3, -787e3, -1327e3, -1210e3)
getz_ice_shelf = (-1624e3, -1130e3, -1234e3, -664e3)
george_vi_ice_shelf = (-2150e3, -1690e3, 540e3, 860e3)
larsen_ice_shelf = (-2430e3, -1920e3, 900e3, 1400e3)
ronne_filchner_ice_shelf = (-1550e3, -500e3, 80e3, 1100e3)
ronne_ice_shelf = (-1550e3, -725e3, 45e3, 970e3)

# automatically created ice shelf regions
abbot_ice_shelf = (-1989e3, -1792e3, -471e3, 51e3)
ainsworth_ice_shelf = (1310e3, 1365e3, -2051e3, -2000e3)
alison_ice_shelf = (-1775e3, -1726e3, 217e3, 278e3)
amery_ice_shelf = (1662e3, 2267e3, 589e3, 860e3)
andreyev_ice_shelf = (931e3, 984e3, -2123e3, -2071e3)
arneb_ice_shelf = (313e3, 356e3, -1918e3, -1874e3)
astrolabe_ice_shelf = (1624e3, 1672e3, -1991e3, -1936e3)
atka_ice_shelf = (-305e3, -201e3, 2060e3, 2155e3)
aviator_ice_shelf = (393e3, 486e3, -1749e3, -1669e3)
bach_ice_shelf = (-1940e3, -1800e3, 533e3, 680e3)
barber_ice_shelf = (610e3, 657e3, -2075e3, -2029e3)
baudouin_ice_shelf = (851e3, 1254e3, 1824e3, 2005e3)
borchgrevink_ice_shelf = (587e3, 894e3, 1917e3, 2153e3)
brahms_ice_shelf = (-1980e3, -1912e3, 540e3, 600e3)
britten_ice_shelf = (-1843e3, -1796e3, 549e3, 596e3)
brunt_stancomb_ice_shelf = (-775e3, -484e3, 1342e3, 1639e3)
campbell_ice_shelf = (426e3, 478e3, -1645e3, -1594e3)
capewashington_ice_shelf = (410e3, 456e3, -1654e3, -1610e3)
cheetham_ice_shelf = (434e3, 490e3, -1512e3, -1463e3)
chugunov_ice_shelf = (587e3, 635e3, -2054e3, -2009e3)
cirque_fjord_ice_shelf = (2112e3, 2159e3, 1268e3, 1322e3)
clarkebay_ice_shelf = (-1512e3, -1468e3, 805e3, 850e3)
commandant_charcot_ice_shelf = (1760e3, 1807e3, -1912e3, -1863e3)
conger_glenzer_ice_shelf = (2549e3, 2654e3, -660e3, -575e3)
cook_ice_shelf = (1010e3, 1159e3, -2153e3, -2038e3)
cosgrove_ice_shelf = (-1816e3, -1721e3, -390e3, -257e3)
crosson_ice_shelf = (-1595e3, -1466e3, -646e3, -509e3)
dalk_ice_shelf = (2169e3, 2220e3, 507e3, 552e3)
dawson_lambton_ice_shelf = (-705e3, -660e3, 1330e3, 1376e3)
deakin_ice_shelf = (1139e3, 1193e3, -2090e3, -2038e3)
dennistoun_ice_shelf = (401e3, 455e3, -2044e3, -1994e3)
dibble_ice_shelf = (1820e3, 1903e3, -1894e3, -1791e3)
dotson_ice_shelf = (-1620e3, -1465e3, -715e3, -584e3)
drury_ice_shelf = (875e3, 927e3, -2132e3, -2081e3)
drygalski_ice_shelf = (380e3, 547e3, -1567e3, -1488e3)
edward_viii_ice_shelf = (2103e3, 2165e3, 1379e3, 1456e3)
ekstrom_ice_shelf = (-382e3, -248e3, 1959e3, 2136e3)
eltanin_bay_ice_shelf = (-1786e3, -1741e3, 254e3, 297e3)
erebus_ice_shelf = (283e3, 334e3, -1328e3, -1283e3)
falkner_ice_shelf = (402e3, 446e3, -1747e3, -1703e3)
ferrigno_ice_shelf = (-1807e3, -1746e3, 173e3, 234e3)
fimbul_ice_shelf = (-132e3, 303e3, 1964e3, 2261e3)
fisher_ice_shelf = (1367e3, 1421e3, -2065e3, -2008e3)
fitzgerald_ice_shelf = (386e3, 450e3, -1769e3, -1702e3)
flatnes_ice_shelf = (2184e3, 2239e3, 496e3, 546e3)
fox_east_ice_shelf = (2360e3, 2405e3, -1108e3, -1060e3)
fox_west_ice_shelf = (-1829e3, -1782e3, 123e3, 169e3)
francais_ice_shelf = (1697e3, 1749e3, -1953e3, -1899e3)
frost_ice_shelf = (1935e3, 2010e3, -1636e3, -1538e3)
gannutz_ice_shelf = (637e3, 684e3, -2076e3, -2028e3)
garfield_ice_shelf = (-1146e3, -1095e3, -1218e3, -1172e3)
geikieinlet_ice_shelf = (442e3, 500e3, -1539e3, -1471e3)
# george_vi_ice_shelf = (-2059e3, -1696e3, 457e3, 831e3)
# getz_ice_shelf = (-1595e3, -1159e3, -1204e3, -694e3)
gillet_ice_shelf = (749e3, 809e3, -2132e3, -2079e3)
hamilton_ice_shelf = (-540e3, -483e3, -1294e3, -1232e3)
hamilton_piedmont_ice_shelf = (-1613e3, -1569e3, -605e3, -562e3)
hannan_ice_shelf = (1787e3, 1851e3, 1626e3, 1707e3)
harbordglacier_ice_shelf = (427e3, 482e3, -1492e3, -1444e3)
harmon_bay_ice_shelf = (-1629e3, -1582e3, -632e3, -587e3)
hayes_coats_coast_ice_shelf = (-722e3, -678e3, 1306e3, 1352e3)
helen_ice_shelf = (2535e3, 2599e3, -207e3, -142e3)
holmes_ice_shelf = (1977e3, 2133e3, -1594e3, -1504e3)
holt_ice_shelf = (-1600e3, -1548e3, -607e3, -555e3)
hornbluff_ice_shelf = (1177e3, 1232e3, -2076e3, -2024e3)
hoseason_ice_shelf = (2106e3, 2169e3, 1303e3, 1355e3)
hovde_ice_shelf = (2196e3, 2242e3, 493e3, 537e3)
hull_ice_shelf = (-1136e3, -1079e3, -1226e3, -1164e3)
hummer_point_ice_shelf = (-1623e3, -1576e3, -615e3, -567e3)
ironside_ice_shelf = (319e3, 372e3, -1950e3, -1896e3)
jackson_ice_shelf = (-1190e3, -1141e3, -1222e3, -1173e3)
jelbart_ice_shelf = (-243e3, -79e3, 2006e3, 2169e3)
kirkby_ice_shelf = (490e3, 537e3, -2079e3, -2035e3)
land_ice_shelf = (-1018e3, -948e3, -1273e3, -1186e3)
larsen_a_ice_shelf = (-2441e3, -2340e3, 1312e3, 1401e3)
larsen_b_ice_shelf = (-2398e3, -2294e3, 1206e3, 1307e3)
larsen_c_ice_shelf = (-2359e3, -2024e3, 962e3, 1315e3)
larsen_d_ice_shelf = (-2135e3, -1591e3, 886e3, 1158e3)
larsen_e_ice_shelf = (-1649e3, -1536e3, 835e3, 923e3)
larsen_f_ice_shelf = (-1572e3, -1482e3, 796e3, 875e3)
larsen_g_ice_shelf = (-1521e3, -1446e3, 767e3, 829e3)
lauritzen_ice_shelf = (888e3, 960e3, -2139e3, -2073e3)
lazarev_ice_shelf = (469e3, 627e3, 2054e3, 2203e3)
lillie_ice_shelf = (540e3, 623e3, -2076e3, -1976e3)
liotard_ice_shelf = (1650e3, 1695e3, -1983e3, -1936e3)
mandible_cirque_ice_shelf = (319e3, 366e3, -1838e3, -1791e3)
manhaul_ice_shelf = (318e3, 363e3, -1919e3, -1875e3)
marin_ice_shelf = (435e3, 482e3, -1471e3, -1425e3)
mariner_ice_shelf = (319e3, 446e3, -1841e3, -1736e3)
marret_ice_shelf = (1720e3, 1772e3, -1946e3, -1901e3)
matusevitch_ice_shelf = (849e3, 912e3, -2141e3, -2073e3)
may_glacier_ice_shelf = (1970e3, 2029e3, -1733e3, -1663e3)
mcleod_ice_shelf = (811e3, 856e3, -2133e3, -2087e3)
mendelssohn_ice_shelf = (-1995e3, -1923e3, 566e3, 646e3)
mertz_ice_shelf = (1377e3, 1486e3, -2159e3, -1956e3)
morse_ice_shelf = (1975e3, 2020e3, -1703e3, -1652e3)
moscow_university_ice_shelf = (2132e3, 2273e3, -1391e3, -1135e3)
moubray_ice_shelf = (304e3, 357e3, -1979e3, -1921e3)
mulebreen_ice_shelf = (2102e3, 2162e3, 1227e3, 1297e3)
myers_ice_shelf = (1886e3, 1947e3, 1592e3, 1642e3)
nansen_ice_shelf = (438e3, 518e3, -1654e3, -1526e3)
# nickerson_ice_shelf = (-989e3, -779e3, -1346e3, -1201e3)
ninnis_ice_shelf = (1212e3, 1345e3, -2059e3, -1973e3)
nivl_ice_shelf = (336e3, 510e3, 2046e3, 2190e3)
noll_ice_shelf = (784e3, 835e3, -2141e3, -2085e3)
nordenskjold_ice_shelf = (419e3, 484e3, -1467e3, -1407e3)
parker_ice_shelf = (403e3, 461e3, -1737e3, -1688e3)
paternostro_ice_shelf = (804e3, 848e3, -2136e3, -2093e3)
perkins_ice_shelf = (-1151e3, -1107e3, -1221e3, -1177e3)
philbin_inlet_ice_shelf = (-1623e3, -1565e3, -737e3, -686e3)
pine_island_ice_shelf = (-1705e3, -1541e3, -384e3, -236e3)
porter_ice_shelf = (1896e3, 1945e3, 1624e3, 1675e3)
pourquoipas_ice_shelf = (1799e3, 1862e3, -1904e3, -1847e3)
prince_harald_ice_shelf = (1257e3, 1425e3, 1734e3, 1942e3)
publications_ice_shelf = (2101e3, 2197e3, 521e3, 625e3)
quar_ice_shelf = (-435e3, -340e3, 1962e3, 2070e3)
quatermain_point_ice_shelf = (315e3, 360e3, -1960e3, -1915e3)
rayner_thyer_ice_shelf = (1793e3, 1873e3, 1586e3, 1664e3)
rennick_ice_shelf = (593e3, 725e3, -2088e3, -1920e3)
richter_ice_shelf = (-608e3, -559e3, -1312e3, -1251e3)
riiser_larsen_ice_shelf = (-658e3, -353e3, 1572e3, 1984e3)
# ronne_filchner_ice_shelf = (-1533e3, -506e3, 115e3, 1059e3)
rose_point_ice_shelf = (-1168e3, -1122e3, -1232e3, -1188e3)
# ross_ice_shelf = (-613e3, 417e3, -1377e3, -414e3)
rund_bay_ice_shelf = (2099e3, 2155e3, 1337e3, 1397e3)
rydberg_ice_shelf = (-1840e3, -1775e3, 313e3, 368e3)
sandford_ice_shelf = (1950e3, 2000e3, -1653e3, -1605e3)
shackleton_ice_shelf = (2506e3, 2765e3, -513e3, -199e3)
shirase_ice_shelf = (1340e3, 1411e3, 1668e3, 1763e3)
skallen_ice_shelf = (1399e3, 1448e3, 1695e3, 1746e3)
slava_ice_shelf = (950e3, 1030e3, -2145e3, -2083e3)
smithinlet_ice_shelf = (420e3, 477e3, -2062e3, -2011e3)
sorsdal_ice_shelf = (2264e3, 2317e3, 453e3, 513e3)
stange_ice_shelf = (-1881e3, -1696e3, 326e3, 499e3)
sulzberger_ice_shelf = (-839e3, -623e3, -1302e3, -1101e3)
suter_ice_shelf = (374e3, 424e3, -1782e3, -1730e3)
suvorov_ice_shelf = (703e3, 770e3, -2102e3, -2040e3)
swinburne_ice_shelf = (-667e3, -588e3, -1286e3, -1188e3)
telen_ice_shelf = (1406e3, 1451e3, 1696e3, 1744e3)
thomson_ice_shelf = (-1823e3, -1768e3, 277e3, 329e3)
thwaites_ice_shelf = (-1627e3, -1496e3, -535e3, -361e3)
tinker_ice_shelf = (424e3, 479e3, -1710e3, -1652e3)
totten_ice_shelf = (2218e3, 2341e3, -1185e3, -984e3)
tracy_tremenchus_ice_shelf = (2564e3, 2664e3, -596e3, -446e3)
tucker_ice_shelf = (304e3, 376e3, -1902e3, -1832e3)
underwood_ice_shelf = (2414e3, 2473e3, -821e3, -759e3)
utsikkar_ice_shelf = (2142e3, 2196e3, 1162e3, 1214e3)
venable_ice_shelf = (-1898e3, -1794e3, 21e3, 156e3)
verdi_ice_shelf = (-1967e3, -1913e3, 511e3, 565e3)
vigrid_ice_shelf = (270e3, 365e3, 2096e3, 2201e3)
vincennes_bay_ice_shelf = (2377e3, 2447e3, -932e3, -802e3)
voyeykov_ice_shelf = (2082e3, 2160e3, -1505e3, -1422e3)
walgreen_ice_shelf = (-1769e3, -1691e3, -386e3, -331e3)
wattbay_ice_shelf = (1454e3, 1507e3, -2068e3, -2013e3)
west_ice_shelf = (2395e3, 2627e3, 34e3, 392e3)
whittle_ice_shelf = (2343e3, 2399e3, -1100e3, -1053e3)
wilkins_ice_shelf = (-2147e3, -1922e3, 554e3, 769e3)
williamson_ice_shelf = (2330e3, 2384e3, -1126e3, -1052e3)
wilmarobertdowner_ice_shelf = (2051e3, 2152e3, 1356e3, 1436e3)
withrow_ice_shelf = (-577e3, -513e3, -1333e3, -1249e3)
wordie_ice_shelf = (-2151e3, -2058e3, 826e3, 928e3)
wylde_ice_shelf = (381e3, 436e3, -1773e3, -1719e3)
zelee_ice_shelf = (1572e3, 1627e3, -2012e3, -1967e3)
zubchatyy_ice_shelf = (1861e3, 1925e3, 1610e3, 1671e3)

# EAST ANTARCTICA
# stancomb_brunt_ice_shelf = ()
# riiser_larsen_ice_shelf = ()
# quar_ice_shelf = ()
# ekstrom_ice_shelf = ()
# atka_ice_shelf = ()
# jelbart_ice_shelf = ()
fimbul_ice_shelf = (-260e3, 430e3, 1900e3, 2350e3)
# vigrid_ice_shelf = ()
# nivl_ice_shelf = ()
# lazarev_ice_shelf = ()
# borchgrevink_ice_shelf = ()
baudouin_ice_shelf = (855e3, 1250e3, 1790e3, 2080e3)
# prince_harald_ice_shelf = ()
# shirase_ice_shelf = ()
# rayner_ice_shelf = ()
# edward_vii_ice_shelf = ()
# wilma_ice_shelf = ()
# robert_ice_shelf = ()
# downer_ice_shelf = ()
amery_ice_shelf = (1530e3, 2460e3, 430e3, 1000e3)
# publications_ice_shelf = ()
# west_ice_shelf = ()
# shackleton_ice_shelf = ()
# tracy_tremenchus_ice_shelf = ()
# conger_ice_shelf = ()
# vicennes_ice_shelf = ()
# totten_ice_shelf = ()
# moscow_university_ice_shelf = ()
# holmes_ice_shelf = ()
# dibble_ice_shelf = ()
# mertz_ice_shelf = ()
# ninnis_ice_shelf = ()
# cook_east_ice_shelf = ()
# rennick_ice_shelf = ()
# lillie_ice_shelf = ()
# mariner_ice_shelf = ()
# aviator_ice_shelf = ()
# nansen_ice_shelf = ()
# drygalski_ice_shelf = ()

# glaciers
# byrd_glacier
# nimrod_glacier
pine_island_glacier = (-1720e3, -1480e3, -380e3, -70e3)
thwaites_glacier = (-1650e3, -1200e3, -600e3, -300e3)
kamb_ice_stream = (-620e3, -220e3, -800e3, -400e3)
# whillans_ice_stream = ()

# seas
ross_sea = (-500e3, 450e3, -2100e3, -1300e3)
# amundsen_sea
# bellinghausen_sea
# weddell_sea

# subglacial lakes
lake_vostok = (1100e3, 1535e3, -470e3, -230e3)
# ice catchements

#####
#####
# Greenland
#####
#####

# regions
greenland = (-700e3, 900e3, -3400e3, -600e3)
north_greenland = (-500e3, 600e3, -1200e3, -650e3)
# northwest_greenland = ()
# northeast_greenland = ()
# west_greenland = ()
# east_greenland = ()
# southeast_greenland = ()
# southwest_greenland = ()

# glaciers
kangerlussuaq_glacier = (380e3, 550e3, -2340e3, -2140e3)


def get_regions() -> dict[str, tuple[float, float, float, float]]:
    """
    get all the regions defined in this module.

    Returns
    -------
    dict[str, tuple[float, float, float, float] ]
        dictionary of each defined region's name and values
    """
    exclude_list = [
        "__",
        "pd",
        "vd",
        "utils",
        "regions",
        "TYPE_CHECKING",
        "Union",
        "maps",
        "ipyleaflet",
        "ipywidgets",
        "Polygon",
        "combine_regions",
        "draw_region",
        "get_regions",
        "annotations",
        "typing",
        "display",
        "alter_region",
        "regions_overlap",
    ]

    return {
        k: v
        for k, v in vars(regions).items()
        if (k not in exclude_list) & (not k.startswith("_"))
    }


def alter_region(
    starting_region: tuple[float, float, float, float],
    zoom: float = 0,
    n_shift: float = 0,
    w_shift: float = 0,
) -> tuple[float, float, float, float]:
    """
    Change a bounding region by shifting the box east/west or north/south, or zooming in
    or out.

    Parameters
    ----------
    starting_region : tuple[float, float, float, float]
        Initial region in meters in format [xmin, xmax, ymin, ymax]
    zoom : float, optional
        zoom in or out, in meters, by default 0
    n_shift : float, optional
        shift north, or south if negative, in meters, by default 0
    w_shift : float, optional
        shift west, or east if negative, in meters, by default 0

    Returns
    -------
    tuple[float, float, float, float]
        Returns the altered region
    """
    starting_e, starting_w = starting_region[0], starting_region[1]
    starting_n, starting_s = starting_region[2], starting_region[3]

    xmin = starting_e + zoom + w_shift
    xmax = starting_w - zoom + w_shift

    ymin = starting_n + zoom - n_shift
    ymax = starting_s - zoom - n_shift

    return (xmin, xmax, ymin, ymax)


def regions_overlap(
    region1: tuple[float, float, float, float],
    region2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Get the overlap of 2 regions.

    Parameters
    ----------
    region1 : tuple[float, float, float, float]
        first region, in the format (xmin, xmax, ymin, ymax)
    region2 : tuple[float, float, float, float]
        second region in the format (xmin, xmax, ymin, ymax)

    Returns
    -------
    tuple[float, float, float, float]
        Overlap  of the 2 supplied regions.
    """

    # create a polygon from the region
    polygon1 = Polygon(
        [
            (region1[0], region1[2]),
            (region1[0], region1[3]),
            (region1[1], region1[3]),
            (region1[1], region1[2]),
            (region1[0], region1[2]),
        ]
    )
    polygon2 = Polygon(
        [
            (region2[0], region2[2]),
            (region2[0], region2[3]),
            (region2[1], region2[3]),
            (region2[1], region2[2]),
            (region2[0], region2[2]),
        ]
    )
    # get the intersection of the 2 polygons
    intersection = polygon1.intersection(polygon2)
    return utils.region_to_bounding_box(intersection.bounds)


def combine_regions(
    region1: tuple[float, float, float, float],
    region2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Get the bounding region of 2 regions.

    Parameters
    ----------
    region1 : tuple[float, float, float, float]
        first region, in the format (xmin, xmax, ymin, ymax)
    region2 : tuple[float, float, float, float]
        second region in the format (xmin, xmax, ymin, ymax)

    Returns
    -------
    tuple[float, float, float, float]
        Bounding region of the 2 supplied regions.
    """
    coords1 = utils.region_to_df(region1)
    coords2 = utils.region_to_df(region2)
    coords_combined = pd.concat((coords1, coords2))
    reg: tuple[float, float, float, float] = vd.get_region(
        (coords_combined.easting, coords_combined.northing)
    )
    return reg


def draw_region(**kwargs: typing.Any) -> list[typing.Any]:
    """
    Plot an interactive map, and use the "Draw a Rectangle" button to draw a rectangle
    and get the bounding region. Vertices will be returned as the output of the
    function.

    Returns
    -------
    list[typing.Any]
        Returns a list of list of vertices for each polyline.

    Example
    -------
    >>> from polartoolkit import regions, utils
    ...
    >>> polygon = regions.draw_region()
    >>> region = utils.polygon_to_region(polygon, hemisphere="north")
    """

    if ipyleaflet is None:
        msg = """
            Missing optional dependency 'ipyleaflet' required for interactive plotting.
        """
        raise ImportError(msg)

    if display is None:
        msg = "Missing optional dependency 'ipython' required for interactive plotting."
        raise ImportError(msg)

    m = maps.interactive_map(**kwargs)

    def clear_m() -> None:
        global poly  # pylint: disable=global-variable-undefined # noqa: PLW0603
        poly = []  # type: ignore[name-defined]

    clear_m()

    mydrawcontrol = ipyleaflet.DrawControl(
        polygon={
            "shapeOptions": {
                "fillColor": "#fca45d",
                "color": "#fca45d",
                "fillOpacity": 0.5,
            }
        },
        polyline={},
        circlemarker={},
        rectangle={},
    )

    def handle_rect_draw(self: typing.Any, action: str, geo_json: typing.Any) -> None:  # noqa: ARG001 # pylint: disable=unused-argument
        global poly  # noqa: PLW0602 # pylint: disable=global-variable-not-assigned
        shapes = []
        for coords in geo_json["geometry"]["coordinates"][0][:-1][:]:
            shapes.append(list(coords))
        shapes = list(shapes)
        if action == "created":
            poly.append(shapes)  # type: ignore[name-defined]

    mydrawcontrol.on_draw(handle_rect_draw)
    m.add_control(mydrawcontrol)

    clear_m()
    display(m)

    return poly  # type: ignore[no-any-return, name-defined]


# Code to define ice shelf regions
# # read into a geodataframe
# ice_shelves = gpd.read_file(fetch.antarctic_boundaries(version="IceShelf"))

# # get list of unique ice shelf names
# names = ice_shelves.NAME.unique()

# # get Series of first words in ice shelf names and number of shelves with that name
# first_words = pd.Series(n.split("_")[0] for n in names).value_counts()

# # get list of lists of ice shelves which should be combined
# shelves_to_combine = {
#     "Ronne_Filchner": ["Ronne", "Filchner"],
# }
# dont_combine = ["Hamilton"]
# for i, j in first_words.items():
#     if (j > 1) & (i not in dont_combine):
#         shelves_to_combine[i] = [n for n in names if n.startswith(i)]

# shelves_to_merge = []
# num_sub_shelves = 0
# for k, v in shelves_to_combine.items():
#     num_sub_shelves += len(v)
#     # merge into new shelf
#     shelf = ice_shelves[ice_shelves.NAME.isin(v)].dissolve()
#     shelf.NAME = k

#     # remove old shelves
#     ice_shelves = ice_shelves[~ice_shelves.NAME.isin(v)]

#     shelves_to_merge.append(shelf)

# # add to dataframe
# ice_shelves = pd.concat([ice_shelves] + shelves_to_merge)

# # there are 2 Fox ice shelves, one in East Antarctica and one in West Antarctica
# # append the region to the name to differentiate
# ice_shelves.loc[
#     (ice_shelves.NAME == "Fox") & (ice_shelves.Regions == "West"), "NAME"
# ] = "Fox_West"
# ice_shelves.loc[
#     (ice_shelves.NAME == "Fox") & (ice_shelves.Regions == "East"), "NAME"
# ] = "Fox_East"

# # sort alphabetically by name
# ice_shelves = ice_shelves.sort_values("NAME").reset_index(drop=True)

# for _i, row in ice_shelves.iterrows():
#     # convert to geodataframe
#     gdf = gpd.GeoDataFrame(row).T.set_geometry("geometry")

#     # define region around each ice shelf with 10km buffer
#     reg = polar_utils.region_to_bounding_box(gdf.iloc[0].geometry.bounds)
#     reg = vd.pad_region(reg, 20e3) # 20 km buffer

#     # round to nearest km
#     reg = tuple([1e3 * round(x / 1e3) for x in reg])

#     # remove last 3 digits
#     reg = tuple([int(x // 1e3) for x in reg])

#     # turn to string
#     reg = f"{gdf.NAME.values[0].lower()}_ice_shelf = ({reg[0]}e3, {reg[1]}e3, {reg[2]}e3, {reg[3]}e3)"

#     print(reg)
