"""
Tests for utils module.
"""

# %%
import deprecation
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyproj
import pytest
import verde as vd
import xarray as xr

from polartoolkit import regions, utils


@deprecation.fail_if_not_removed
def test_alter_region():
    utils.alter_region(regions.ross_ice_shelf, zoom=10e3)


def dummy_grid() -> xr.Dataset:
    (x, y, z) = vd.grid_coordinates(
        region=(-100, 100, 200, 400),
        spacing=100,
        extra_coords=20,
    )

    # create topographic features
    misfit = y**2

    return vd.make_xarray_grid(
        (x, y),
        (misfit, z),
        data_names=("misfit", "upward"),
        dims=("northing", "easting"),
    )


# %%
def test_subset_grid():
    "test the subset grid function"
    grid = dummy_grid()

    region = (0, 100, 200, 300)
    subset = utils.subset_grid(grid.misfit, region)

    reg = utils.get_grid_info(subset)[1]

    assert reg == region


def test_subset_grid_bigger():
    "test the subset grid function with a region bigger than the grid"
    grid = dummy_grid()

    region = (-200, 100, 200, 500)
    subset = utils.subset_grid(grid.misfit, region)

    reg = utils.get_grid_info(subset)[1]

    assert reg == (-100, 100, 200, 400)


def test_rmse():
    """
    test the RMSE function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMSE
    rmse = utils.rmse(data)
    # test that the RMSE is correct
    assert rmse == pytest.approx(2.160247, rel=0.0001)


def test_rmse_median():
    """
    test the RMedianSE function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMedianSE
    rmse = utils.rmse(data, as_median=True)
    # test that the RMSE is correct
    assert rmse == 2


def test_get_grid_info():
    """
    test the get_grid_info function
    """
    grid = dummy_grid()

    info = utils.get_grid_info(grid.misfit)

    assert info == (100.0, (-100.0, 100.0, 200.0, 400.0), 40000.0, 160000.0, "g")


def test_dd2dms():
    """
    test the dd2dms function
    """

    dd = 130.25

    dms = utils.dd2dms(dd)

    assert dms == "130:15:0.0"


def test_region_to_df():
    """
    test the region_to_df function
    """

    reg = regions.ross_ice_shelf

    df = utils.region_to_df(reg)

    expected = pd.DataFrame(
        {
            "easting": [
                -680000.0,
                470000.0,
                -680000.0,
                470000.0,
            ],
            "northing": [
                -1420000.0,
                -1420000.0,
                -310000.0,
                -310000.0,
            ],
        }
    )

    pd.testing.assert_frame_equal(df, expected)

    reg2 = utils.region_to_df(df, reverse=True)

    assert reg2 == reg


def test_region_xy_to_ll_north():
    """
    test the GMT_xy_to_ll function
    """

    reg_xy = regions.north_greenland

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="north", dms=True)

    assert reg_ll == (
        "-82:34:6.931303778954316",
        "-2:17:26.19615349869855",
        "77:39:39.3112071675132",
        "82:26:25.183721040550154",
    )

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="north")

    assert reg_ll == (
        -82.56859202882748,
        -2.2906100426385274,
        77.66091977976876,
        82.44032881140015,
    )


def test_region_xy_to_ll_south():
    """
    test the GMT_xy_to_ll function
    """

    reg_xy = regions.ross_ice_shelf

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="south", dms=True)

    assert reg_ll == (
        "-154:24:41.35269126086496",
        "161:41:10.261402247124352",
        "-84:49:17.300473876937758",
        "-75:34:58.96941344602965",
    )

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="south")

    assert reg_ll == (
        -154.41148685868356,
        161.6861837228464,
        -84.8214723538547,
        -75.58304705929056,
    )


def test_region_to_bounding_box():
    """
    test the region_to_bounding_box function
    """

    reg = regions.ross_ice_shelf

    box = utils.region_to_bounding_box(reg)

    assert box == (-680000.0, -1420000.0, 470000.0, -310000.0)


def test_latlon_to_epsg3031():
    """
    test the latlon_to_epsg3031 function
    """

    df_ll = pd.DataFrame(
        {
            "lon": [-154.411487, 161.686184, -114.507405, 123.407825],
            "lat": [-75.583047, -76.296586, -83.129754, -84.82147],
        }
    )

    df_xy = utils.latlon_to_epsg3031(df_ll)

    expected = pd.DataFrame(
        {
            "easting": [
                -680000.0,
                470000.0,
                -680000.0,
                470000.0,
            ],
            "northing": [
                -1420000.0,
                -1420000.0,
                -310000.0,
                -310000.0,
            ],
        }
    )
    pd.testing.assert_frame_equal(df_xy[["easting", "northing"]], expected)


def test_latlon_to_epsg3031_region():
    """
    test the latlon_to_epsg3031 function output a region
    """

    df_ll = pd.DataFrame(
        {
            "lon": [-154.411487, 161.686184, -114.507405, 123.407825],
            "lat": [-75.583047, -76.296586, -83.129754, -84.82147],
        }
    )

    reg = utils.latlon_to_epsg3031(df_ll, reg=True)

    assert reg == pytest.approx(regions.ross_ice_shelf, abs=10)


def test_epsg3031_to_latlon():
    """
    test the epsg3031_to_latlon function
    """

    df_xy = utils.region_to_df(regions.ross_ice_shelf)

    df_ll = utils.epsg3031_to_latlon(df_xy)

    expected = pd.DataFrame(
        {
            "easting": [
                -680000.0,
                470000.0,
                -680000.0,
                470000.0,
            ],
            "northing": [
                -1420000.0,
                -1420000.0,
                -310000.0,
                -310000.0,
            ],
            "lon": [-154.411487, 161.686184, -114.507405, 123.407825],
            "lat": [-75.583047, -76.296586, -83.129754, -84.82147],
        }
    )

    pd.testing.assert_frame_equal(df_ll, expected)


def test_epsg3031_to_latlon_region():
    """
    test the epsg3031_to_latlon function output a region
    """

    df_xy = utils.region_to_df(regions.ross_ice_shelf)

    reg = utils.epsg3031_to_latlon(df_xy, reg=True)

    assert reg == pytest.approx((-154.41, 161.69, -84.82, -75.58), abs=0.01)


def test_latlon_to_epsg3413():
    """
    test the latlon_to_epsg3413 function
    """

    df_ll = pd.DataFrame(
        {
            "lon": [-35.776078, -31.773128, -34.930937, -30.586402],
            "lat": [68.366181, 68.070993, 70.129560, 69.806264],
        }
    )

    df_xy = utils.latlon_to_epsg3413(df_ll)

    expected = pd.DataFrame(
        {
            "easting": [
                380000.0,
                550000.0,
                380000.0,
                550000.0,
            ],
            "northing": [
                -2340000.0,
                -2340000.0,
                -2140000.0,
                -2140000.0,
            ],
        }
    )
    pd.testing.assert_frame_equal(df_xy[["easting", "northing"]], expected)


def test_latlon_to_epsg3413_region():
    """
    test the latlon_to_epsg3413 function output a region
    """

    df_ll = pd.DataFrame(
        {
            "lon": [-35.776078, -31.773128, -34.930937, -30.586402],
            "lat": [68.366181, 68.070993, 70.129560, 69.806264],
        }
    )

    reg = utils.latlon_to_epsg3413(df_ll, reg=True)

    assert reg == pytest.approx(regions.kangerlussuaq_glacier, abs=10)


def test_epsg3413_to_latlon():
    """
    test the epsg3413_to_latlon function
    """

    df_xy = utils.region_to_df(regions.kangerlussuaq_glacier)

    df_ll = utils.epsg3413_to_latlon(df_xy)

    expected = pd.DataFrame(
        {
            "easting": [
                380000.0,
                550000.0,
                380000.0,
                550000.0,
            ],
            "northing": [
                -2340000.0,
                -2340000.0,
                -2140000.0,
                -2140000.0,
            ],
            "lon": [-35.776078, -31.773128, -34.930937, -30.586402],
            "lat": [68.366181, 68.070993, 70.129560, 69.806264],
        }
    )
    pd.testing.assert_frame_equal(df_ll, expected)


def test_epsg3413_to_latlon_region():
    """
    test the epsg3413_to_latlon function output a region
    """

    df_xy = utils.region_to_df(regions.kangerlussuaq_glacier)

    reg = utils.epsg3413_to_latlon(df_xy, reg=True)

    assert reg == pytest.approx((-35.78, -30.59, 68.07, 70.13), abs=0.01)


def test_points_inside_region():
    """
    test the points_inside_region function
    """
    # first point is inside, second is outside
    df = pd.DataFrame(
        {
            "easting": [-50e3, 0],
            "northing": [-1000e3, 0],
        }
    )

    assert len(df) == 2

    reg = regions.ross_ice_shelf

    df_in = utils.points_inside_region(df, reg)

    assert len(df_in) == 1
    assert df_in.easting.iloc[0] == -50e3

    df_out = utils.points_inside_region(df, reg, reverse=True)

    assert len(df_out) == 1
    assert df_out.easting.iloc[0] == 0.0


def test_normalize_rotation():
    """
    test wrapping rotation angles into the range (-180, 180]
    """
    assert utils.normalize_rotation(0) == 0.0
    assert utils.normalize_rotation(45) == 45.0
    assert utils.normalize_rotation(-45) == -45.0
    # both ends of the range map to +180
    assert utils.normalize_rotation(180) == 180.0
    assert utils.normalize_rotation(-180) == 180.0
    # wrapping
    assert utils.normalize_rotation(360) == 0.0
    assert utils.normalize_rotation(190) == -170.0
    assert utils.normalize_rotation(-190) == 170.0
    # must not produce -0.0, which would render as "-0" in a GMT projection string
    assert not np.signbit(utils.normalize_rotation(0))
    assert not np.signbit(utils.normalize_rotation(360))


def test_set_proj_unrotated_unchanged():
    """
    an unrotated figure must produce exactly the historic projection strings
    """
    region = regions.ross_ice_shelf

    _, proj_latlon, _, _ = utils.set_proj(region, epsg="3031", fig_height=10)
    assert proj_latlon is not None
    assert proj_latlon.startswith("s0/-90/-71/")

    _, proj_latlon, _, _ = utils.set_proj(region, epsg="3413", fig_height=10)
    assert proj_latlon is not None
    assert proj_latlon.startswith("s-45/90/70/")


def test_set_proj_rotated():
    """
    rotating only shifts the central meridian, and the shift is in opposite directions
    for the two hemispheres
    """
    region = regions.ross_ice_shelf

    # south polar: lon_0 = 0 - rotation
    proj, proj_latlon, width, height = utils.set_proj(
        region, epsg="3031", fig_height=10, rotation=45
    )
    assert proj_latlon is not None
    assert proj_latlon.startswith("s-45/-90/-71/")

    # north polar: lon_0 = -45 + rotation
    _, proj_latlon, _, _ = utils.set_proj(
        region, epsg="3413", fig_height=10, rotation=45
    )
    assert proj_latlon is not None
    assert proj_latlon.startswith("s0/90/70/")

    # the linear projection and the figure size are unaffected by rotation
    proj0, _, width0, height0 = utils.set_proj(region, epsg="3031", fig_height=10)
    assert proj == proj0
    assert (width, height) == (width0, height0)


def test_set_proj_rotation_requires_polar_stereographic():
    """
    rotation is only meaningful for a projection centred on the pole
    """
    with pytest.raises(NotImplementedError, match="polar stereographic"):
        utils.set_proj(regions.ross_ice_shelf, epsg="3995", fig_height=10, rotation=45)


def central_meridian(crs: pyproj.CRS) -> float:
    """
    read a CRS's central meridian without going via a PROJ string

    `CRS.to_dict()` / `to_proj4()` emit a UserWarning about losing projection
    information, which the test suite escalates to an error.
    """
    return float(
        next(
            p.value
            for p in crs.coordinate_operation.params
            if p.name == "Longitude of origin"
        )
    )


def test_rotated_crs():
    """
    test the rotated CRS only differs from the base CRS by its central meridian
    """
    # south polar subtracts the rotation, north polar adds it
    for epsg, expected in (("3031", -45.0), ("3413", 0.0)):
        assert central_meridian(utils.rotated_crs(epsg, 45)) == pytest.approx(expected)
        # a zero rotation must reproduce the base projection's central meridian
        base = utils.POLAR_STEREOGRAPHIC_PARAMS[epsg][2]
        assert central_meridian(utils.rotated_crs(epsg, 0)) == pytest.approx(base)


def test_rotated_crs_zero_rotation_is_identity():
    """
    a zero rotation must leave coordinates untouched
    """
    easting = np.array([1e5, -3e5, 0.0])
    northing = np.array([-9e5, 2e5, -1.5e6])

    for epsg in ("3031", "3413"):
        got_e, got_n = utils.rotation_transformer(epsg, 0)(easting, northing)
        npt.assert_allclose(got_e, easting, atol=1e-6)
        npt.assert_allclose(got_n, northing, atol=1e-6)


def test_rotated_central_meridian():
    """
    test the central meridian shift, including wrapping past +/-180
    """
    assert utils.rotated_central_meridian("3031", 0) == 0.0
    assert utils.rotated_central_meridian("3031", 45) == -45.0
    assert utils.rotated_central_meridian("3413", 0) == -45.0
    assert utils.rotated_central_meridian("3413", 45) == 0.0
    # results stay within (-180, 180]
    for epsg in ("3031", "3413"):
        for rotation in (-350, -170, 0, 95, 170, 350):
            lon_0 = utils.rotated_central_meridian(epsg, rotation)
            assert -180 < lon_0 <= 180

    with pytest.raises(NotImplementedError, match="polar stereographic"):
        utils.rotated_central_meridian("3857", 45)


def test_rotation_transformer_roundtrip():
    """
    rotating into the rotated frame and back must return the original coordinates
    """
    easting = np.array([1e5, -3e5, 0.0, 7e5])
    northing = np.array([-9e5, 2e5, -1.5e6, 4e5])

    for epsg in ("3031", "3413"):
        forward = utils.rotation_transformer(epsg, 33.5)
        inverse = utils.rotation_transformer(epsg, 33.5, inverse=True)
        back_e, back_n = inverse(*forward(easting, northing))
        npt.assert_allclose(back_e, easting, atol=1e-6)
        npt.assert_allclose(back_n, northing, atol=1e-6)


def test_rotation_is_clockwise():
    """
    a positive rotation must move the map clockwise, in both hemispheres

    A point on the base projection's central meridian lies straight up the page for
    EPSG:3031 and straight down for EPSG:3413. After a clockwise rotation of `angle` its
    bearing from page-up must have increased by exactly `angle`.
    """
    angle = 45.0
    for epsg in ("3031", "3413"):
        # a point on the base central meridian
        lat_0, _, lon_0 = utils.POLAR_STEREOGRAPHIC_PARAMS[epsg]
        lat = lat_0 - 10 if lat_0 > 0 else lat_0 + 10
        easting, northing = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        ).transform(lon_0, lat)
        rotated = utils.rotation_transformer(epsg, angle)(easting, northing)

        before = np.degrees(np.arctan2(easting, northing))
        after = np.degrees(np.arctan2(*rotated))
        swing = (after - before + 180) % 360 - 180
        assert swing == pytest.approx(angle, abs=1e-6)


def test_rotation_matches_rotation_matrix():
    """
    the CRS-based rotation must equal a plain clockwise rotation about the pole
    """
    rng = np.random.default_rng(0)
    easting = rng.uniform(-2e6, 2e6, 200)
    northing = rng.uniform(-2e6, 2e6, 200)
    angle = 45.0

    # plain clockwise rotation matrix about (0, 0)
    theta = np.radians(-angle)
    expected_e = easting * np.cos(theta) - northing * np.sin(theta)
    expected_n = easting * np.sin(theta) + northing * np.cos(theta)

    for epsg in ("3031", "3413"):
        got_e, got_n = utils.rotation_transformer(epsg, angle)(easting, northing)
        npt.assert_allclose(got_e, expected_e, atol=1e-6)
        npt.assert_allclose(got_n, expected_n, atol=1e-6)


def test_rotate_region():
    """
    rotating a region expands it to contain the whole of the original
    """
    region = regions.ross_ice_shelf

    assert utils.rotate_region(region, 0, epsg="3031") == region

    rotated = utils.rotate_region(region, 45, epsg="3031")

    # every corner of the original must fall inside the rotated region
    easting, northing = utils.rotation_transformer("3031", 45)(
        *utils.region_corners(region)
    )
    assert vd.inside((easting, northing), rotated).all()

    # a w x h region becomes (w|cos| + h|sin|) by (w|sin| + h|cos|)
    width, height = region[1] - region[0], region[3] - region[2]
    cos, sin = abs(np.cos(np.radians(45))), abs(np.sin(np.radians(45)))
    assert rotated[1] - rotated[0] == pytest.approx(width * cos + height * sin)
    assert rotated[3] - rotated[2] == pytest.approx(width * sin + height * cos)

    # a quarter turn just swaps the dimensions
    quarter = utils.rotate_region(region, 90, epsg="3031")
    assert quarter[1] - quarter[0] == pytest.approx(height)
    assert quarter[3] - quarter[2] == pytest.approx(width)

    # a half turn leaves the size alone
    half = utils.rotate_region(region, 180, epsg="3031")
    assert half[1] - half[0] == pytest.approx(width)
    assert half[3] - half[2] == pytest.approx(height)


def test_rotate_region_never_clips():
    """
    no matter the rotation or hemisphere, the original region stays fully visible

    This is the property that stops a rotated map cutting off data the user asked for.
    """
    region = regions.ross_ice_shelf

    for epsg in ("3031", "3413"):
        for rotation in (0, 15, 45, 90, 137, 180, -60):
            rotated = utils.rotate_region(region, rotation, epsg=epsg)
            easting, northing = utils.rotation_transformer(epsg, rotation)(
                *utils.region_corners(region)
            )
            assert vd.inside((easting, northing), rotated).all()


def test_region_corners():
    """
    test the closed corner ring used to draw a region as a polygon
    """
    easting, northing = utils.region_corners((-1, 2, -3, 4))

    # closed ring
    assert len(easting) == len(northing) == 5
    assert easting[0] == easting[-1]
    assert northing[0] == northing[-1]
    # matches the axis-aligned rectangle it replaces
    npt.assert_array_equal(easting, [-1, -1, 2, 2, -1])
    npt.assert_array_equal(northing, [-3, 4, 4, -3, -3])


def test_native_top_longitude():
    """
    test which line of longitude is at the top of an unrotated map

    EPSG:3031 is drawn with the Greenwich meridian up, but EPSG:3413 with 135 degrees
    east up - which is why Greenland, near -45 degrees, sits at the bottom of the page.
    """
    assert utils.native_top_longitude("3031") == 0.0
    assert utils.native_top_longitude("3413") == 135.0

    with pytest.raises(NotImplementedError, match="polar stereographic"):
        utils.native_top_longitude("3857")


def test_top_longitude_to_rotation():
    """
    test converting a desired top longitude into a clockwise rotation
    """
    # no rotation needed to keep the native orientation
    for epsg in ("3031", "3413"):
        native = utils.native_top_longitude(epsg)
        assert utils.top_longitude_to_rotation(epsg, native) == 0.0

    # the two hemispheres turn in opposite senses
    assert utils.top_longitude_to_rotation("3031", -45) == 45.0
    assert utils.top_longitude_to_rotation("3413", 180) == 45.0

    with pytest.raises(NotImplementedError, match="polar stereographic"):
        utils.top_longitude_to_rotation("3857", 45)


def test_top_longitude_round_trip():
    """
    the requested longitude must actually end up at the top of the page

    Converts to a rotation, builds the rotated CRS, then asks which meridian is now
    parallel to the page's vertical axis.
    """
    for epsg in ("3031", "3413"):
        lat_0 = utils.POLAR_STEREOGRAPHIC_PARAMS[epsg][0]
        for top_longitude in (0, 45, 90, 135, 180, -90, -33.5):
            rotation = utils.top_longitude_to_rotation(epsg, top_longitude)

            # a point straight up the page from the pole, in the rotated frame
            lon, _ = pyproj.Transformer.from_crs(
                utils.rotated_crs(epsg, rotation), "EPSG:4326", always_xy=True
            ).transform(0.0, 1_000_000.0)

            assert utils.normalize_rotation(lon - top_longitude) == pytest.approx(
                0.0, abs=1e-6
            )

            # and it agrees with the central meridian relation
            expected = utils.normalize_rotation(
                utils.rotated_central_meridian(epsg, rotation)
                + (180.0 if lat_0 > 0 else 0.0)
            )
            assert utils.normalize_rotation(expected - top_longitude) == pytest.approx(
                0.0, abs=1e-9
            )


def test_unrotated_region():
    """
    test the region of data needed to cover a rotated map
    """
    region = regions.ross_ice_shelf

    # no rotation needs nothing extra
    assert utils.unrotated_region(region, epsg="3031") == region
    assert utils.unrotated_region(region, 0, epsg="3031") == region
    assert utils.unrotated_region(region, 0, epsg="3413") == region

    needed = utils.unrotated_region(region, 45, epsg="3031")

    # the tilted footprint needs a bigger box than the region itself
    assert needed[0] < region[0]
    assert needed[1] > region[1]
    assert needed[2] < region[2]
    assert needed[3] > region[3]

    # the plotted region already grew to (w+h)/sqrt(2) square at 45 degrees, and filling
    # that in unrotated coordinates needs a further factor of sqrt(2), so (w + h)
    width, height = region[1] - region[0], region[3] - region[2]
    expected = width + height
    assert needed[1] - needed[0] == pytest.approx(expected, rel=1e-6)
    assert needed[3] - needed[2] == pytest.approx(expected, rel=1e-6)

    # a half turn needs no extra data at all
    npt.assert_allclose(
        utils.unrotated_region(region, 180, epsg="3031"), region, atol=1e-6
    )


def test_unrotated_region_covers_the_footprint():
    """
    every corner of the rotated map must fall inside the returned region
    """
    region = regions.ross_ice_shelf

    for epsg in ("3031", "3413"):
        for rotation in (0, 30, 45, 90, 180, -75):
            needed = utils.unrotated_region(region, rotation, epsg=epsg)

            # the plotted footprint, converted back to unrotated coordinates
            plotted = utils.rotate_region(region, rotation, epsg=epsg)
            easting, northing = utils.rotation_transformer(
                epsg, rotation, inverse=True
            )(*utils.region_corners(plotted))

            assert vd.inside((easting, northing), needed).all()


def test_rotation_to_top_longitude():
    """
    test reporting which line of longitude ends up at the top of the page
    """
    # with no rotation this is the projection's native orientation
    for epsg in ("3031", "3413"):
        assert utils.rotation_to_top_longitude(epsg, 0) == utils.native_top_longitude(
            epsg
        )

    # the two hemispheres turn in opposite senses
    assert utils.rotation_to_top_longitude("3031", 45) == -45.0
    assert utils.rotation_to_top_longitude("3413", 45) == 180.0

    with pytest.raises(NotImplementedError, match="polar stereographic"):
        utils.rotation_to_top_longitude("3857", 45)


def test_top_longitude_rotation_round_trip():
    """
    the two conversions must invert each other, in both directions
    """
    for epsg in ("3031", "3413"):
        for value in (0, 30, 45, 90, 180, -75, -33.5):
            assert utils.top_longitude_to_rotation(
                epsg, utils.rotation_to_top_longitude(epsg, value)
            ) == pytest.approx(utils.normalize_rotation(value))
            assert utils.rotation_to_top_longitude(
                epsg, utils.top_longitude_to_rotation(epsg, value)
            ) == pytest.approx(utils.normalize_rotation(value))


def tilted_box(tilt: float, width: float = 600e3, height: float = 250e3):
    """A rectangle of known size, rotated anticlockwise by `tilt` degrees."""
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [-width / 2, height / 2],
            [width / 2, height / 2],
            [width / 2, -height / 2],
            [-width / 2, -height / 2],
        ]
    )
    angle = np.radians(tilt)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return corners @ rotation_matrix.T + [-1.4e6, -9.5e5]


def test_region_is_corners():
    """
    test telling a plain region apart from corner coordinates
    """
    assert not utils._region_is_corners((-1, 2, -3, 4))
    assert not utils._region_is_corners([-1, 2, -3, 4])
    assert not utils._region_is_corners(None)

    assert utils._region_is_corners(tilted_box(30))
    assert utils._region_is_corners((np.array([0, 1, 2]), np.array([0, 1, 2])))
    assert utils._region_is_corners(
        pd.DataFrame({"easting": [0, 1], "northing": [0, 1]})
    )


def test_corners_to_region():
    """
    test the bounding region of corner coordinates
    """
    corners = tilted_box(0)
    region = utils._corners_to_region(corners, "3031")
    assert region[1] - region[0] == pytest.approx(600e3)
    assert region[3] - region[2] == pytest.approx(250e3)


def test_oriented_region_recovers_the_tilt():
    """
    the inferred rotation must square a tilted box up exactly

    Whatever angle a box was built at, rotating by the inferred rotation must recover
    its true width and height rather than a larger bounding box.
    """
    width, height = 600e3, 250e3

    for tilt in (0, 20, 40, -25, 70, 115, 89, -89):
        corners = tilted_box(tilt, width, height)
        box = utils.oriented_region(corners, epsg="3031")
        rotation, easting, northing = box.rotation, box.easting, box.northing

        rotated = vd.get_region(
            utils.rotation_transformer("3031", rotation)(easting, northing)
        )
        assert rotated[1] - rotated[0] == pytest.approx(width, rel=1e-6)
        assert rotated[3] - rotated[2] == pytest.approx(height, rel=1e-6)


def test_oriented_region_puts_north_up():
    """
    of the two half turns which square a box up, the north-up one must be chosen

    In the rotated frame the pole sits at the origin, so at the box centre north points
    away from it in the south and towards it in the north.
    """
    for epsg, centre in (("3031", (-1.4e6, -9.5e5)), ("3413", (-3e5, -2.2e6))):
        lat_0 = utils.POLAR_STEREOGRAPHIC_PARAMS[epsg][0]
        for tilt in (0, 20, 40, -25, 70, 115, 89, -89):
            corners = tilted_box(tilt) - [-1.4e6, -9.5e5] + list(centre)
            box = utils.oriented_region(corners, epsg=epsg)

            middle = np.column_stack([box.easting, box.northing])[:-1].mean(axis=0)
            _, northing = utils.rotation_transformer(epsg, box.rotation)(*middle)
            assert np.sign(lat_0) * northing < 0


def test_oriented_region_beats_axis_aligned():
    """
    an oriented box must never be larger than the axis-aligned one
    """
    for tilt in (0, 15, 30, 45, 60, 80):
        corners = tilted_box(tilt)
        box = utils.oriented_region(corners, epsg="3031")
        rotation, easting, northing = box.rotation, box.easting, box.northing

        oriented = vd.get_region(
            utils.rotation_transformer("3031", rotation)(easting, northing)
        )
        axis_aligned = utils._corners_to_region(corners, "3031")

        oriented_area = (oriented[1] - oriented[0]) * (oriented[3] - oriented[2])
        aligned_area = (axis_aligned[1] - axis_aligned[0]) * (
            axis_aligned[3] - axis_aligned[2]
        )
        assert oriented_area <= aligned_area * (1 + 1e-9)


def test_oriented_region_accepts_a_dataframe():
    """
    test the coordinate-input handling
    """
    corners = tilted_box(30)
    frame = pd.DataFrame({"easting": corners[:, 0], "northing": corners[:, 1]})

    from_array = utils.oriented_region(corners, epsg="3031").rotation
    from_frame = utils.oriented_region(frame, epsg="3031").rotation
    from_pair = utils.oriented_region(
        (corners[:, 0], corners[:, 1]), epsg="3031"
    ).rotation

    assert from_frame == pytest.approx(from_array)
    assert from_pair == pytest.approx(from_array)


def test_oriented_region_returns_everything_needed():
    """
    the returned box must carry the plotting region and the region to fetch data for
    """
    corners = tilted_box(35)
    box = utils.oriented_region(corners, epsg="3031")

    # the plotting region is the tight box, in the rotated frame
    assert box.region[1] - box.region[0] == pytest.approx(600e3, rel=1e-6)
    assert box.region[3] - box.region[2] == pytest.approx(250e3, rel=1e-6)

    # the data region is axis-aligned in the unrotated projection and contains the box
    assert vd.inside((box.easting, box.northing), box.data_region).all()
    assert box.data_region[1] - box.data_region[0] > box.region[1] - box.region[0]


def test_oriented_region_pad():
    """
    padding zooms out along the box's own axes, not the unrotated ones
    """
    corners = tilted_box(35)
    tight = utils.oriented_region(corners, epsg="3031")
    padded = utils.oriented_region(corners, epsg="3031", pad=50e3)

    # same orientation, 50 km wider on every side
    assert padded.rotation == pytest.approx(tight.rotation)
    assert padded.region[1] - padded.region[0] == pytest.approx(600e3 + 100e3, rel=1e-6)
    assert padded.region[3] - padded.region[2] == pytest.approx(250e3 + 100e3, rel=1e-6)

    # a tuple pads the two axes separately, following verde's (north, east) order
    uneven = utils.oriented_region(corners, epsg="3031", pad=(25e3, 100e3))
    assert uneven.region[1] - uneven.region[0] == pytest.approx(600e3 + 200e3, rel=1e-6)
    assert uneven.region[3] - uneven.region[2] == pytest.approx(250e3 + 50e3, rel=1e-6)

    # negative padding zooms in
    zoomed = utils.oriented_region(corners, epsg="3031", pad=-25e3)
    assert zoomed.region[1] - zoomed.region[0] == pytest.approx(600e3 - 50e3, rel=1e-6)
