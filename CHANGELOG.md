# Changelog
Automatically updated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/en/latest/)
with commit parsing of [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).

## Unreleased
### 🐛 Bug Fixes
* fixing release ([`7de9a24`](https://github.com/mdtanker/polartoolkit/commit/7de9a2427a628bf2dc7e5f4f8bb481c9317535a4))
* fix release ([`cf57090`](https://github.com/mdtanker/polartoolkit/commit/cf57090a3e7a752b0fbb52e04b1f81da5b182b76))
* still fixing ([`ecf44c5`](https://github.com/mdtanker/polartoolkit/commit/ecf44c58e9827183484859677849a311ed5eb1fc))
* still fixing release ([`4bf136f`](https://github.com/mdtanker/polartoolkit/commit/4bf136f5022a95d9a348ade526c7cc055900793e))
* still fixing release action ([`027d9cf`](https://github.com/mdtanker/polartoolkit/commit/027d9cf1c67910d745451ccdd7e0e404cded8ea0))
* fixing release GHA ([`ce5e7dd`](https://github.com/mdtanker/polartoolkit/commit/ce5e7dd4f64df2e2060ff9f4c011d716f12ff04c))
* get PSR to upload to pypi ([`156b3d2`](https://github.com/mdtanker/polartoolkit/commit/156b3d21b51b1f8c5646549e9767301e529a12a9))
* update python semantic release ([`269c7d2`](https://github.com/mdtanker/polartoolkit/commit/269c7d2ea19cdd6332d4ad21070bcfb3b436f394))
### 🧰 Chores / Maintenance
* increase deprecations to v2.0.0 ([`f5ac4b3`](https://github.com/mdtanker/polartoolkit/commit/f5ac4b363ef3e21b70eb279a92c480fd90c8292f))


## v1.0.0 (2025-06-04)
### 🐛 Bug Fixes
* update fetch tests ([`3218acf`](https://github.com/mdtanker/polartoolkit/commit/3218acf1d66fd7dac434614239b8b135795d22ea))
* update region tests ([`68d8ebc`](https://github.com/mdtanker/polartoolkit/commit/68d8ebcce772e8911484d5c2632664d2d1ec8d59))
* remove basal_melt fetch due to changed data access ([`8a8f64c`](https://github.com/mdtanker/polartoolkit/commit/8a8f64c3952aad92d4c19830f59378f63f2e912b))
* remove unneeded checks in `resample_grid` ([`3a73b32`](https://github.com/mdtanker/polartoolkit/commit/3a73b323b394cbe00a8e546ecc3352a6d6a95abd))
* bug in `maps.subplots` ([`b966565`](https://github.com/mdtanker/polartoolkit/commit/b966565d006c1e7c6e3d05ad3b00a7d72454f264))
* use pygmt.grdsample in `grd_compare` to ensure grids can be subtracted ([`c1f9b92`](https://github.com/mdtanker/polartoolkit/commit/c1f9b92e0bbfa753cfaf7a3135918d11e4e88af5))
* bug where coordinates displayed on clicking `interactive_map` were incorrect ([`494ecc1`](https://github.com/mdtanker/polartoolkit/commit/494ecc1fb40d1638808e96f663ca48edb8eabe7b))
* remove bug where `layers_version` wasn't being used in `plot_profiles` ([`2f4f118`](https://github.com/mdtanker/polartoolkit/commit/2f4f11890cf8fd19f3d0a859d7f6e9a55ff49057))
* bug in distance calculation for profiles, and rename relative and cumulative distance functions ([`788045d`](https://github.com/mdtanker/polartoolkit/commit/788045d6584a7f0a4a96eed7c57e18d508b84587))
* check region and spacing provide if no grid in `utils.mask_from_polygon` ([`2567780`](https://github.com/mdtanker/polartoolkit/commit/2567780a9284d46dffcfe8fe6bfefe4b477e09a1))
* still fill layers in profile if entirely above plot ([`e850a05`](https://github.com/mdtanker/polartoolkit/commit/e850a0591af1d467ada6c400c76c9df543e29d90))
* check cpt_lims for cbar_end_triangles ([`061e939`](https://github.com/mdtanker/polartoolkit/commit/061e93974d08622aa4527a395f59d136cfb09417))
* plotting code should be under if plot condition for `grd_compare` ([`05daa80`](https://github.com/mdtanker/polartoolkit/commit/05daa8077da0b83c01193f0867ce672883125586))
### 📦️ Build
* add pyogrio as dep and remove geopandas upper version limit ([`9ffd5ce`](https://github.com/mdtanker/polartoolkit/commit/9ffd5ce5f5ddc96e16f1ea3f2047f03ee3f8e378))
### 🧰 Chores / Maintenance
### ✏️ Formatting
* minor changes to fetch.py ([`52b49c2`](https://github.com/mdtanker/polartoolkit/commit/52b49c2c9f95036a6e541e6d2066c50195791aeb))
* fixes for new pre-commit hook versions ([`5b534ce`](https://github.com/mdtanker/polartoolkit/commit/5b534ce949f9611f333ed336c01a80c964b07614))
* pre-commit fixes ([`b49b4a0`](https://github.com/mdtanker/polartoolkit/commit/b49b4a0f4c72f5ab154b66000bfb3d04b9c2149b))
* pre-commit fixes ([`e2e3516`](https://github.com/mdtanker/polartoolkit/commit/e2e35166e0421a2b0f4c9708f0ff1551325d5909))
### 📖 Documentation
* minor fixes to docs ([`217ff31`](https://github.com/mdtanker/polartoolkit/commit/217ff31317382b49ab7e9ec1ca621b577f3998b8))
* rerun all docs ([`5fe7619`](https://github.com/mdtanker/polartoolkit/commit/5fe7619fe691afc65d59224e56a8056b3ae9ff07))
* minor docstring fixes ([`28bdeb8`](https://github.com/mdtanker/polartoolkit/commit/28bdeb836b9d051c344d1577faa98d8e191fafb0))
### 🚀 Features
* add kwarg `cpt_log` to make colormap on a log scale. ([`e8406b6`](https://github.com/mdtanker/polartoolkit/commit/e8406b628da835701face2b47f26bcc24fbc23bc))
* add `regions_overlap` function ([`e8ddbeb`](https://github.com/mdtanker/polartoolkit/commit/e8ddbebd78e420c64beff792be2ec8d1d05fb778))
* ensure ice surface, water surface, and earth surface are in correct order in `plot_profiles` ([`e668a90`](https://github.com/mdtanker/polartoolkit/commit/e668a90857fc9d4cdda07bb1339fc022462c2123))
* add option to choose modis cmap: `modis_cmap` ([`29ad0ac`](https://github.com/mdtanker/polartoolkit/commit/29ad0ac3484e6e0b582383fc5f34622fe3d142c7))
* allow drapegrid for `plot_3d` ([`3f095c1`](https://github.com/mdtanker/polartoolkit/commit/3f095c1ee410ad937301d31f879d2ee318096457))
* enable absolute values option for colormaps ([`159de8b`](https://github.com/mdtanker/polartoolkit/commit/159de8b4c3aecc211374abbc249d15528bd2a7ac))
* allow automatic and chosen colorbar end triangles ([`77f3547`](https://github.com/mdtanker/polartoolkit/commit/77f35475087d7e7272fde20285843452ccc6a94c))
* allow specifying colorbar height percentage ([`d327304`](https://github.com/mdtanker/polartoolkit/commit/d327304ae5e2420dd7a0ca3d31442156ff2cc9d6))
###  🎨 Refactor
* use zarr instead of nc for some preprocessed fetch calls ([`03c89b0`](https://github.com/mdtanker/polartoolkit/commit/03c89b0df62fdc8f6f19d2aab85c08f10fb5e688))
* various changes to fetch functions ([`45837a9`](https://github.com/mdtanker/polartoolkit/commit/45837a9474dff2431c79575dd7364970d3181375))
* remove all initial region, spacing, and registrations from fetch functions ([`b9c4203`](https://github.com/mdtanker/polartoolkit/commit/b9c420365768fe1166682290f50d0e2a166666a4))
* `interactive_map` zoom level based on basemap type ([`06125f5`](https://github.com/mdtanker/polartoolkit/commit/06125f5d798063248745acaf3099292fd77f5417))
* for southern hemisphere, change default interactive map basemap to `Imagery` to allow zooming in ([`ac11744`](https://github.com/mdtanker/polartoolkit/commit/ac11744237db3ce8a48f2f30315e09f920893747))
* remove `show` kwarg of `interactive_map` ([`2507c8f`](https://github.com/mdtanker/polartoolkit/commit/2507c8f414f18422b0e6163eb72eadff26f27082))
* remove `clip` kwarg from `plot_profiles` and `plot_data` and do distance clipping before grd sampling ([`2f3bed9`](https://github.com/mdtanker/polartoolkit/commit/2f3bed97c074529282a736f1a20259fd9301c072))
* require extracting grid info in `resample_grid` and reorganize function ([`a38b94b`](https://github.com/mdtanker/polartoolkit/commit/a38b94bd52f6d7c6c54fa22ecc147b5f14e47da2))
* deprecate `xr_grid` and `grid_file` parameters for `utils.mask_from_shp` and use `grid` instead ([`9a2af43`](https://github.com/mdtanker/polartoolkit/commit/9a2af430200df560343cbae487a682ef5ce5b24e))
* use bedmap2 as default for profiles for speed ([`41dcb5a`](https://github.com/mdtanker/polartoolkit/commit/41dcb5a2c22f61ddd4c69ed72bc4d2aff82b71ab))
* clean up some code and add some debug logging ([`9240bc8`](https://github.com/mdtanker/polartoolkit/commit/9240bc822cca855b1730436f8a0530c9ebeba846))

## v0.9.1 (2025-04-07)
### 🐛 Bug Fixes
* update bedmap3 fetch with after registration issue ([`6cf33ed`](https://github.com/mdtanker/polartoolkit/commit/6cf33ed7a3cb1ccd7e159ca53fc915b09d0b2b4a))
* update some fetch hashes ([`f6e4ab7`](https://github.com/mdtanker/polartoolkit/commit/f6e4ab7a3c93c56263aa886a7b9be7373b2332e8))
* use updated coord names ([`35feba1`](https://github.com/mdtanker/polartoolkit/commit/35feba1622ad9d174ea7b9665d99440d7db74c98))
* resample grids if samping and reg are same, but registration is different ([`1c1636a`](https://github.com/mdtanker/polartoolkit/commit/1c1636acb20753d7f9f97c24cdf00f01cda268fb))
### 🧰 Chores / Maintenance
* reduce sig figs on test_gravity ([`1005757`](https://github.com/mdtanker/polartoolkit/commit/1005757d335f6fbb4fb8e54003a783021ee83751))
* update test_bedmap3 ([`6f69a34`](https://github.com/mdtanker/polartoolkit/commit/6f69a34f1e78cd275f556e0c7fe7a23c1b3e7748))
* update values for test_bedmap_points ([`0eec659`](https://github.com/mdtanker/polartoolkit/commit/0eec6591610d774614d7036e91bb1d2aad537a0a))
* add filterwarnings to some tests ([`b0636e7`](https://github.com/mdtanker/polartoolkit/commit/b0636e7dbabdbcd00b2e585585fbe52dd682c47c))
###  🎨 Refactor
* change order of some code for clarity ([`9e315b0`](https://github.com/mdtanker/polartoolkit/commit/9e315b0ba5df47b97e7bd70b2b41213d46a162f7))

## v0.9.0 (2025-03-31)
### 🐛 Bug Fixes
* more informative errors raised in fetch ([`a94704c`](https://github.com/mdtanker/polartoolkit/commit/a94704c78d52f8080a6b68ccaf1ffcfa494894de))
* warn in grd2cpt called for points not grid ([`9b11476`](https://github.com/mdtanker/polartoolkit/commit/9b114766886d1819cda6622293f032f41db63e40))
* if region not specified and subplotting use existing figure region ([`d7a8b83`](https://github.com/mdtanker/polartoolkit/commit/d7a8b838fa3f0bed4e1cde97e41d7a892dd3c754))
* unnecessary kwargs passed to add_colorbar ([`b4302c4`](https://github.com/mdtanker/polartoolkit/commit/b4302c46c8920835810ccb29980c999a0f2fd20e))
* pass region to add_inset in 'plot_grd' and 'basemap' ([`958154c`](https://github.com/mdtanker/polartoolkit/commit/958154c428cb9e6badaa45fe4eacc10374e35ca1))
* subset GHF points by region during fetch ([`92a41ad`](https://github.com/mdtanker/polartoolkit/commit/92a41ad4cdd1d6e96b518a273dfd94f36fd0a24a))
* add defaults for coordinate conversion output column names ([`66a649b`](https://github.com/mdtanker/polartoolkit/commit/66a649b392c8f762944c5f198c94d75898b6208a))
* use assumed coordinate names in reprojections ([`423ddc5`](https://github.com/mdtanker/polartoolkit/commit/423ddc56fdd094ebb19a5c837bb52ef30ffece72))
### 🧰 Chores / Maintenance
* update backup env yml ([`510d0c6`](https://github.com/mdtanker/polartoolkit/commit/510d0c6cbb1efb55cf84479949bfbf867bacdd38))
* update make remove comand ([`d4805b4`](https://github.com/mdtanker/polartoolkit/commit/d4805b4e978de08ba86ef33823948fa8f3c3f1ca))
* add logging to maps.py ([`13930be`](https://github.com/mdtanker/polartoolkit/commit/13930bee63ba749a70390d6efbe62015af0013af))
### ✏️ Formatting
* quotes error ([`d5fe11a`](https://github.com/mdtanker/polartoolkit/commit/d5fe11a7d13ccd08be27717dc5b5fe42ef32310d))
### 📖 Documentation
* rerun all docs ([`e24cd70`](https://github.com/mdtanker/polartoolkit/commit/e24cd702aef719acb398d83e32eca0b82125745b))
* add some missing docstrings ([`95aa34b`](https://github.com/mdtanker/polartoolkit/commit/95aa34b4b0d8979732386a7fffa716e8172099d4))
### 🚀 Features
* add bedmap3 grids ([`3bab405`](https://github.com/mdtanker/polartoolkit/commit/3bab40587dda4bf98bce6049006e94ad09a31f95))
* allow extra x and y shift amounts ([`5bef9ca`](https://github.com/mdtanker/polartoolkit/commit/5bef9ca785a03762be9dafcee35cef25a7e9a39c))
* allow specifying cbar histogram height with 'cbar_hist_height', independent from 'cbar_yoffset ([`02ae70d`](https://github.com/mdtanker/polartoolkit/commit/02ae70d9d2c32d0f55ee4a747ac8de4940a8c006))
* allow specifing simple basemap colors and pen ([`c711274`](https://github.com/mdtanker/polartoolkit/commit/c711274a80a86d10d15ccffbc689ab2e7dd79bee))
* allow setting number of columns in layers legend in profiles with 'layers_legend_columns' ([`149fbf7`](https://github.com/mdtanker/polartoolkit/commit/149fbf7ca49a9eaf7ca1fa8b141560fa09d48729))
* add verbose kwarg for 'add_box' and 'add_colorbar' ([`92a476d`](https://github.com/mdtanker/polartoolkit/commit/92a476d4e2e1ae636d167e59f161683d55401a6d))
* allow specifying robust percentiles in various functions ([`f1b62f1`](https://github.com/mdtanker/polartoolkit/commit/f1b62f1896c037cbd0f4b2ea9957985c0d216a25))
* add padding options and more filter types to 'filter_grid' ([`425aea8`](https://github.com/mdtanker/polartoolkit/commit/425aea84b135a0aacb38e05719af44d26698bceb))
###  🎨 Refactor
* use easting and northing throughout profiles module ([`82df4f6`](https://github.com/mdtanker/polartoolkit/commit/82df4f6b8ab69a1026651fa23b59a301342b9549))
* use 'easting' and 'northing' instead of 'x' and 'y' throughout code ([`856e054`](https://github.com/mdtanker/polartoolkit/commit/856e054b3eb2b964111a96cf56570cbd4ff14de3))
* bedmap3 as default for profiles ([`b2fc5d6`](https://github.com/mdtanker/polartoolkit/commit/b2fc5d6e1c36763bd51e4de19eb37f0b29b14017))
* determine y shift amount based on what is being plotted. ([`901be27`](https://github.com/mdtanker/polartoolkit/commit/901be2762f51ade9bc38bff0ff48a7409fdd7a78))
* set title and frames at beginning of plotting ([`f2131ab`](https://github.com/mdtanker/polartoolkit/commit/f2131abdc1a34c083430a0028503d4633d5c76ea))
* use seperate transparency kwarg for frame and grid ([`d9ca761`](https://github.com/mdtanker/polartoolkit/commit/d9ca761a1331fe248e7aca0448011d06fdc52663))
* use easting and northing for bedmap points ([`2ce8510`](https://github.com/mdtanker/polartoolkit/commit/2ce851066d006a6b443128ee7b11c0d1bafeafe9))
* use warnings instead of logging for some cases ([`2876442`](https://github.com/mdtanker/polartoolkit/commit/287644287b3217fd6575bd6b0383acfa2ddd4c42))
* specify point fill with column name ([`cd59881`](https://github.com/mdtanker/polartoolkit/commit/cd59881e192e006e62aab29b965e4ffa65876f07))
* change 'scale_font_color' to 'scalebar_font_color' ([`18e7b84`](https://github.com/mdtanker/polartoolkit/commit/18e7b8462a09c0854d23ad28f67d8f8f7264b690))
* use measures-v2 groundingline by default instead of depoorter ([`07ad14b`](https://github.com/mdtanker/polartoolkit/commit/07ad14bee052dbcb301021e749b94b1b094e436a))
*  rename 'inset_pos' to 'inset_position' and remove 'inset_offset' ([`52a537f`](https://github.com/mdtanker/polartoolkit/commit/52a537f86154a06ade079f9e4d80a2492adca822))
* scalebar and inset length by default relative to shortest dimension ([`d6acd76`](https://github.com/mdtanker/polartoolkit/commit/d6acd7695adc96cdaf1a0f3f76e4a11f31af4396))
* change kwarg 'scale_length_perc' to 'scalebar_length_perc' ([`4b77be3`](https://github.com/mdtanker/polartoolkit/commit/4b77be38a5709c62eb6f4ccf008128dfc549f76f))
* change `grd_compare` defaults to automatically plot and no inset ([`9d0041b`](https://github.com/mdtanker/polartoolkit/commit/9d0041b0b6c831d9bbdaad24df21dd84fac5e7bc))

## v0.8.1 (2025-02-19)
### 🐛 Bug Fixes
* update from angular to conventional commits to fix deprecation warning ([`079a5db`](https://github.com/mdtanker/polartoolkit/commit/079a5db21fa9a8c4bcab8caba93c1083d452b370))
### 🧰 Chores / Maintenance
* update python-semantic-release version ([`80ecf5a`](https://github.com/mdtanker/polartoolkit/commit/80ecf5a6d7e351666b5d9638d14285ca0923d9ce))
* update semantic release changelog template ([`ecd6834`](https://github.com/mdtanker/polartoolkit/commit/ecd6834e031d5fba2f61635699a0c61d83360a1d))
* wait to deprecate functions till v1.0.0 ([`ffae0de`](https://github.com/mdtanker/polartoolkit/commit/ffae0de55c248052b60c860bd55463398a8c98b5))
* fix some tests ([`c33aadb`](https://github.com/mdtanker/polartoolkit/commit/c33aadb98b53d2eb44aa29207e59f8ade0406d16))
* update test marks ([`3b15338`](https://github.com/mdtanker/polartoolkit/commit/3b153389fb0efaa0fcc98913cf67a2071270f6f5))
### ✏️ Formatting
* empty line ([`defa73e`](https://github.com/mdtanker/polartoolkit/commit/defa73e37eb3edd06460cdada810550ecc6cb6f3))
###  🎨 Refactor
* add kwargs for legend labels for coasts and points ([`20c4d0e`](https://github.com/mdtanker/polartoolkit/commit/20c4d0e5c51f5698ac308afd99b17d701939fd54))

## v0.8.0 (2025-02-19)
### 🐛 Bug Fixes
* update to new fetch gravity function ([`82d0024`](https://github.com/mdtanker/polartoolkit/commit/82d0024533e7b9f4a894cccccef14dcbe04c9fae))
* fix typos in fetch functions ([`0b18528`](https://github.com/mdtanker/polartoolkit/commit/0b18528ebcd2e8557fac19822dc275fb90f290bf))
* reorder columns in testing reprojection functions ([`a4d82dd`](https://github.com/mdtanker/polartoolkit/commit/a4d82dde76ea89a0c83ff398c6a03b537a66b3ed))
* minor fix in `maps` ([`9866e72`](https://github.com/mdtanker/polartoolkit/commit/9866e72dae7e67084db0b587ddebd7a3516130c5))
* fix issue with reprojecting IBCSO coverage data ([`214bf6f`](https://github.com/mdtanker/polartoolkit/commit/214bf6fef3091a96b93f50d7eac3af638dacb0d7))
* allow many subplot labels ([`3a130cf`](https://github.com/mdtanker/polartoolkit/commit/3a130cfc104fda7957abd9815c639b283fc4cf0e))
* allow single geometry in `mask_from_shp` ([`c231c77`](https://github.com/mdtanker/polartoolkit/commit/c231c77eafad789f41289df09d37a561b77180a7))
* allow showing all columns for points in `interactive data` ([`0692a6b`](https://github.com/mdtanker/polartoolkit/commit/0692a6b114df9eccd9e8641dfb5bbc59cde46069))
### 📦️ Build
* unpin unnecessary verde version ([`c9a5c1d`](https://github.com/mdtanker/polartoolkit/commit/c9a5c1d78b5336d6e202760c6f6bd9e6c9cb0c22))
### 🧰 Chores / Maintenance
* disable disallow-caps pre-commit to fix issue ([`3cfe9a8`](https://github.com/mdtanker/polartoolkit/commit/3cfe9a8914225a2f151afaa116833f1c8b9bf738))
* mark tests with fetch and earthdata marks ([`ace576e`](https://github.com/mdtanker/polartoolkit/commit/ace576eaf0c0d850bd689174625c56bf438ba670))
* fix tests ([`2453f47`](https://github.com/mdtanker/polartoolkit/commit/2453f47b1507d92ef8e6536bbf20f347ada7cb58))
* update broken fetch tests ([`8fbe37c`](https://github.com/mdtanker/polartoolkit/commit/8fbe37c5712a4240a34cbfccbaee885c069e343a))
* add tests to profiles module ([`52db507`](https://github.com/mdtanker/polartoolkit/commit/52db50717353cd5ea0ab0ba62e46bf0df6bd27af))
* add tests for regions module ([`9a230fc`](https://github.com/mdtanker/polartoolkit/commit/9a230fcfc9fce163dcad66f3e07fcb2ae47e797e))
* warn if dataset provided instead of dataarray ([`fd090af`](https://github.com/mdtanker/polartoolkit/commit/fd090affb48b32fb8c39f41313f3c4aae46db448))
* add debugging logs ([`a88b76f`](https://github.com/mdtanker/polartoolkit/commit/a88b76f5e1a082440c211587eee49c5834d3ad1d))
* make command fix ([`207535a`](https://github.com/mdtanker/polartoolkit/commit/207535a7abc99eed527598c3ce1ac857e6825999))
* update backup env to `v0.7.1` ([`ef1f621`](https://github.com/mdtanker/polartoolkit/commit/ef1f621cbb8bc9bf1dc8e0429e026ab2f372594f))
### ✏️ Formatting
* pre-commit fixes ([`9487733`](https://github.com/mdtanker/polartoolkit/commit/9487733539a0b59bc7858365c2de5a88d1bf4381))
* pre-commit fixes ([`a166181`](https://github.com/mdtanker/polartoolkit/commit/a166181abe323e6bde7fd7c651d6f17cc51812d3))
* type checking fixes ([`f1e7ede`](https://github.com/mdtanker/polartoolkit/commit/f1e7ede6795bcb35259b7b63989cb2e2e8b2add3))
### Other
* add setuptools to RTD_env to fix issue ([`ee5178a`](https://github.com/mdtanker/polartoolkit/commit/ee5178a03aab4589a92ade2e14597fcd37381402))
### 📖 Documentation
* use new fetch gravity function ([`281028e`](https://github.com/mdtanker/polartoolkit/commit/281028e24a17f6fd2c426c08a76eee5db0440363))
* update simple map tutorial ([`269147a`](https://github.com/mdtanker/polartoolkit/commit/269147a0a82ac93887cbabe49e3a3acdac4ef255))
* fix gravity dataset gallery ([`123f57e`](https://github.com/mdtanker/polartoolkit/commit/123f57e01799dd8bc3e4ca740c9500389b1489d3))
* update ibcso coverage notebook ([`d4a8ebb`](https://github.com/mdtanker/polartoolkit/commit/d4a8ebb31c82fde233ae921d878a121b365eb7ff))
* update bedmap points gallery ([`69fd85d`](https://github.com/mdtanker/polartoolkit/commit/69fd85dc5a08f2271018ffab1a15f48671dfd4e5))
* fixes to some docstring ([`aa7017e`](https://github.com/mdtanker/polartoolkit/commit/aa7017e3ea9d2a09ceeb3b2aa78121625ecb3e7c))
### 🚀 Features
* add range of basemap options ([`86d2e0d`](https://github.com/mdtanker/polartoolkit/commit/86d2e0d14a428ccbd20d211fa17e8c85ae9285b8))
* add region for getz ice shelf ([`467f5b8`](https://github.com/mdtanker/polartoolkit/commit/467f5b84cdb7ed187846c58830142751a59ebdcd))
* allow resampling during fetching of `mass_change` and `basal_melt` ([`db1f464`](https://github.com/mdtanker/polartoolkit/commit/db1f464f277e9a17fd178329623bf1ed0f403a40))
* add kwargs to all fetch calls to pass to `resample_grid` ([`9e60600`](https://github.com/mdtanker/polartoolkit/commit/9e60600c1b92946d36a4bca44487decb4b157edf))
* add `add_simple_basemap` plotting function ([`247c3aa`](https://github.com/mdtanker/polartoolkit/commit/247c3aac41e08c0d10b80832d249c7a421191af4))
* add `points_inside_shp` function ([`e859569`](https://github.com/mdtanker/polartoolkit/commit/e8595696a383a557afca7052dbc7205afac3bd02))
* add fetch for bedmap 2 and 3 point data ([`7cca1a0`](https://github.com/mdtanker/polartoolkit/commit/7cca1a0f2b33c5a8c81f6e8e89d1cb6a5bede91d))
* add kwargs to pass to `resample_grd` ([`0cf71d7`](https://github.com/mdtanker/polartoolkit/commit/0cf71d75e88f231a27da323ded57ac55c14734d2))
* add ice shelf buttressing dataset to fetch ([`3e6f8b1`](https://github.com/mdtanker/polartoolkit/commit/3e6f8b17bb43d3a88bd2be666b64ff7b01e7d6b8))
* allow specifying individual points, reverse_cpt, inset, scalebar for subplots ([`49e53f9`](https://github.com/mdtanker/polartoolkit/commit/49e53f9314bebd1adfbfdc5f79b2921ea2c7c693))
* allow subplot row and column titles ([`dad1ce5`](https://github.com/mdtanker/polartoolkit/commit/dad1ce5ac71929fd78e9b38767f07ec202f8315f))
* allow setting transparency in `basemap` ([`6a18fb8`](https://github.com/mdtanker/polartoolkit/commit/6a18fb8485447ebd2a9859981a99c524acf5d726))
* add absolute value option and robust percentiles option to `get_min_max` ([`490198c`](https://github.com/mdtanker/polartoolkit/commit/490198c9b19e38982178328360a4259315b24639))
* add `filter_grid` function for filtering grids, add xrft as dependency ([`c2b2696`](https://github.com/mdtanker/polartoolkit/commit/c2b2696726d182f6e4821fcef90170189c3bd7d2))
* add `nearest_grid_fill` for interpolating grids ([`80d5f10`](https://github.com/mdtanker/polartoolkit/commit/80d5f106e37b531a7bd00369f45512a0871c0e8e))
* add cpt_lims kwarg to `grd_compare` ([`31e4735`](https://github.com/mdtanker/polartoolkit/commit/31e4735802f70b1a93ca81c7401fe85b14d3c57b))
* allow both grids and points in `get_min_max` and `get_combined_min_max` ([`290fdc5`](https://github.com/mdtanker/polartoolkit/commit/290fdc5b8d92c5eea8b97aafdf6c45466d9081a4))
* all cmap for points in `basemap` ([`908f04b`](https://github.com/mdtanker/polartoolkit/commit/908f04bb730662c9da1d1fc3cd8e2ac8a354d50e))
* allow creation of cmap from point values as well as grids values with new `points` parameter to `set_cmap` ([`fa01f54`](https://github.com/mdtanker/polartoolkit/commit/fa01f54a35d1385ac1cf7db2036bb875d99c17b6))
* allow choice of vertical reference and cleanup `fetch.ibcso` ([`333a373`](https://github.com/mdtanker/polartoolkit/commit/333a3738dfb05e6c12774b717e6c94d8a2df1a8f))
###  🎨 Refactor
* change from `variable` to `version` in `buttressing` and `basal_melt` to be consistent ([`56af29d`](https://github.com/mdtanker/polartoolkit/commit/56af29d88f8d174a970012f60f31a4eeb7ef4b32))
* remove compression on zarr files for simplicity ([`8c8dd5b`](https://github.com/mdtanker/polartoolkit/commit/8c8dd5bdb55db69cfb7e8c8aa7fed2f55f0a0005))
* setup and use logger instance for polartoolkit ([`ff1a02a`](https://github.com/mdtanker/polartoolkit/commit/ff1a02a9217e86e0a20332164ad60477d3d1d4b0))
* update gravity fetch ([`d67d6b7`](https://github.com/mdtanker/polartoolkit/commit/d67d6b77164c21aee5dc7c82080348121d827d65))
* drop antgg-update and make other gravity fetches return datasets with elevation and error grids ([`3be3a83`](https://github.com/mdtanker/polartoolkit/commit/3be3a83c27f1383a50f54aa61e08469ffa788087))
* use consistent import of pathlib ([`2bfd3ef`](https://github.com/mdtanker/polartoolkit/commit/2bfd3ef8e432cdd98e212582aed8aa6d6deedb05))
* use tuples instead of lists where applicable ([`6bc1f20`](https://github.com/mdtanker/polartoolkit/commit/6bc1f204d322a86832ed19fe715ce24556f65763))
* combine and simplify reprojection functions ([`6f3ac05`](https://github.com/mdtanker/polartoolkit/commit/6f3ac05ffa67b0300f3ca5397ed88db4071d2257))
* check and use pyogrio for reading coastlines ([`9702e4d`](https://github.com/mdtanker/polartoolkit/commit/9702e4db0d59d26e184e3feb72dc066ba2f8523e))
* automatically increase vertical shift between plots if using colorbar histogram ([`0f0820c`](https://github.com/mdtanker/polartoolkit/commit/0f0820cf211cab3b5c614e0ef05765b27611508e))
* clean up `fetch.ibcso_coverage` ([`41da27f`](https://github.com/mdtanker/polartoolkit/commit/41da27f812bba15f62ca5d3118f6e00845b5ac1e))
* check if pyogrio is installed and use if it is, warn if its not ([`576db88`](https://github.com/mdtanker/polartoolkit/commit/576db88740ae51d70e7e5dd389774f7202a6d962))

## v0.7.1 (2024-12-17)
### 🐛 Bug Fixes
* update `maps.subplots()` to be more user friendly ([`1f3bc35`](https://github.com/mdtanker/polartoolkit/commit/1f3bc353151639afdc3d72f00a64fe348c3e05e0))
* raise error if EarthData login doesn't work ([`1576c00`](https://github.com/mdtanker/polartoolkit/commit/1576c009352fe5dadeba7ea7c636b028a7dd4f65))
* make region parameter of `ibcso_coverage` options ([`0064c05`](https://github.com/mdtanker/polartoolkit/commit/0064c0582907a7c3f3f4a47eee17825b7361fe24))
* more special cases with `square_subplots` ([`466eb56`](https://github.com/mdtanker/polartoolkit/commit/466eb56a3b9df8f7271c75afffb54b068af65d54))
* raise error if EarthData login fails ([`4f6762d`](https://github.com/mdtanker/polartoolkit/commit/4f6762d1e94f6ca836cf53b2fec59b3f8e07b151))
* bug with `plot_profile` and `plot_data` ([`23bde34`](https://github.com/mdtanker/polartoolkit/commit/23bde342e19dfd155ac21c5703458834586f0ba2))
* extract previous fig_height for subplots ([`cb9e93f`](https://github.com/mdtanker/polartoolkit/commit/cb9e93f148479fc407e47ebd21e2150c60085f88))
* re-add test for deepbedmap ([`c825e67`](https://github.com/mdtanker/polartoolkit/commit/c825e670359383321f06e98633eac7eec5468be8))
* bad link in fetch `mass_change` ([`54b67d1`](https://github.com/mdtanker/polartoolkit/commit/54b67d1606fd7ff86ca319cc01de78633b623f34))
### 🧰 Chores / Maintenance
* update run docs commands ([`533dd69`](https://github.com/mdtanker/polartoolkit/commit/533dd694a62f90cffd8fcc02f2e40c4ba94c081c))
* dev install with environment.yml ([`74b9afd`](https://github.com/mdtanker/polartoolkit/commit/74b9afdc8d6bf996d4c5740cf740682236ef530b))
* remove codespell ignore-regrex ([`6b6affc`](https://github.com/mdtanker/polartoolkit/commit/6b6affc8611abe539c452262b972b5b3d7a5e9fb))
* remove unused `raps` and `coherency` functions ([`315f562`](https://github.com/mdtanker/polartoolkit/commit/315f56229f11a4b30bb40c8482c563bc910da3d7))
* clean up .py license headers ([`4853222`](https://github.com/mdtanker/polartoolkit/commit/4853222db398670928885303e3c5d4cad41a8005))
* update backup yml ([`a8f7809`](https://github.com/mdtanker/polartoolkit/commit/a8f7809387e25d215d070ea1884c7d271cdeab89))
* add tests for fetch `groundingline` and `antarctic_boundaries` ([`ff7d9a2`](https://github.com/mdtanker/polartoolkit/commit/ff7d9a2a2ccd30b9469e74d742be010512831e96))
### ✏️ Formatting
* pre-commit fixes ([`522f18e`](https://github.com/mdtanker/polartoolkit/commit/522f18e7d8544f53ee1ac9a8a7fd3d10f45a0676))
* ruff fixes ([`70e59a1`](https://github.com/mdtanker/polartoolkit/commit/70e59a11a70d764e52e3eaad0d59bfc26e838256))
### 📖 Documentation
* update install instructions ([`c2e9845`](https://github.com/mdtanker/polartoolkit/commit/c2e9845b16a6fa968d63816367b1403ca4812934))
* update and rerun all docs ipynb ([`0e8e6eb`](https://github.com/mdtanker/polartoolkit/commit/0e8e6eb4af8d0380fe8ad875c14dce6c389c97d0))
###  🎨 Refactor
* reduce default number of cbar histogram bins ([`75543fb`](https://github.com/mdtanker/polartoolkit/commit/75543fbc02de9272ffab4f85d7f9dba1c3aea0b3))
* reduce default cbar y offset ([`04cd3ed`](https://github.com/mdtanker/polartoolkit/commit/04cd3ed42796e360f107ee40d88efd14ced63c20))
* change default vertical origin shift to down instead of up ([`e3a9e3c`](https://github.com/mdtanker/polartoolkit/commit/e3a9e3c4f1c9f2763e4c2eb7829cffdd80c152b2))
* need to specify `ais` or `gris` in `fetch.mass_change` ([`26ae3d6`](https://github.com/mdtanker/polartoolkit/commit/26ae3d64ce4bda1376ad26393da95cbe2bd8b03c))
* remove pyogrio dependency and replace with geopandas ([`3b4c315`](https://github.com/mdtanker/polartoolkit/commit/3b4c315cf4fa870416ebe40990c808a85d1daaa3))
* remove matplotlib plotting options, and matplotib and seaborn dependencies ([`b0458c9`](https://github.com/mdtanker/polartoolkit/commit/b0458c9b96f8ec885c6ffbeebf4f34a9b0107c2d))

## v0.7.0 (2024-12-03)
### 🐛 Bug Fixes
* broken shading parameter in `plot_grd` ([`97ec19b`](https://github.com/mdtanker/polartoolkit/commit/97ec19b243a570176b5cb44a6a71a9466aee4979))
* bug in `mask_from_shp` ([`b3e3dbc`](https://github.com/mdtanker/polartoolkit/commit/b3e3dbcf4d3513ec2c9013c2a46920c59673b567))
* raise warning if passing buffer to `alter_region` ([`be16120`](https://github.com/mdtanker/polartoolkit/commit/be16120628eb84b68f62e57d1943787381008fe9))
* bug in `square_subplots` ([`861310c`](https://github.com/mdtanker/polartoolkit/commit/861310ca886edb7ad3b534e8371136744fdc89ff))
* allow easting and northing as coordinate column names as backups for x and y in `sample_grids` ([`8cb9f8a`](https://github.com/mdtanker/polartoolkit/commit/8cb9f8a36cb359d395a2ea7fb1e3953344b3fee2))
* check for valid version strings in `fetch.mass_change` ([`24cd984`](https://github.com/mdtanker/polartoolkit/commit/24cd984953c08aa753b45714c9f72e6f04d34d7f))
### 🧰 Chores / Maintenance
* update make test install commands and instructions ([`c236a9e`](https://github.com/mdtanker/polartoolkit/commit/c236a9e15ef78febb3cf7559587b6bc8a9c97c6c))
* update backup environmental.yml for `v0.6.0` ([`5626d22`](https://github.com/mdtanker/polartoolkit/commit/5626d221d22fc7af8e6bb2985d10abc21d42344d))
### ✏️ Formatting
* pre-commit fixes ([`1082e32`](https://github.com/mdtanker/polartoolkit/commit/1082e32db7eb799562d512a861db7ae6021cf3de))
* pre-commit fixes ([`4d9be74`](https://github.com/mdtanker/polartoolkit/commit/4d9be74566def222d6794eb20aa8b7a83716a212))
### 📖 Documentation
* remove overall dataset gallery ([`99dadb2`](https://github.com/mdtanker/polartoolkit/commit/99dadb2ef29dbbefefdf9ce4bc09cf1d3df727d7))
* fixes to some docstrings ([`bf2b747`](https://github.com/mdtanker/polartoolkit/commit/bf2b747021aa17216ed4b690618442dbcc7f1bfa))
* rerun notebook ([`37edf1d`](https://github.com/mdtanker/polartoolkit/commit/37edf1dcd6abeb8e139bad6b583c63cad91f0295))
### 🚀 Features
* add EIGEN gravity data for Arctic ([`a1b1943`](https://github.com/mdtanker/polartoolkit/commit/a1b19433a59731c5851aa56d6e68ef01be5744f8))
* add ice velocity basemap to `interactive_map` ([`7f410f8`](https://github.com/mdtanker/polartoolkit/commit/7f410f8b99480d6517ac8bed7e5a2f623b0e9a91))
* allow easting and northing as backups for x and y for coord column names in various functions ([`8f8ae84`](https://github.com/mdtanker/polartoolkit/commit/8f8ae84053963fac3390ef773cf318a00e8b264c))
* add LCS-1 satellite magnetic anomaly grids ([`3b400f9`](https://github.com/mdtanker/polartoolkit/commit/3b400f9e659a05bcbb800d359c9e79c51e0311fa))

## v0.6.0 (2024-11-18)
### 📦️ Build
* update backup env yml for`v0.5.1` ([`7a1b984`](https://github.com/mdtanker/polartoolkit/commit/7a1b98434fed49a94d097a77557252a5a5314dce))
### 🧰 Chores / Maintenance
* remove ruff commands and use pre-commit ([`e73a310`](https://github.com/mdtanker/polartoolkit/commit/e73a3100a21a39bde2c5d2d048c815713603cdca))
### ✏️ Formatting
* pre-commit fixes ([`7514e3a`](https://github.com/mdtanker/polartoolkit/commit/7514e3a80dd919e322953faa40b274294070f981))
### 📖 Documentation
* rerun docs ([`fba71d7`](https://github.com/mdtanker/polartoolkit/commit/fba71d7e5350ab6425ada571dc50c42b0d8306dc))
* add commit message convention to contribute guide ([`959f6f4`](https://github.com/mdtanker/polartoolkit/commit/959f6f49abe2da3c8404fb8354e671630057a7b8))
* add JOSS to citing page ([`54d8273`](https://github.com/mdtanker/polartoolkit/commit/54d827391ac988f14070c37fe12730d7c58427bd))
* fix issue in contrib guide ([`f776f03`](https://github.com/mdtanker/polartoolkit/commit/f776f03d56c581683688d27552eaec71f4b59029))
### 🚀 Features
* add ANTGG2021 gravity compilation ([`f782a85`](https://github.com/mdtanker/polartoolkit/commit/f782a8551a34c1f01f2fa877690ccf1fb7cede3b))
* allow setting y limits in profile plots ([`894fafb`](https://github.com/mdtanker/polartoolkit/commit/894fafb10bdcebac6f4ff13323888c6f0bdf0182))

## v0.5.1 (2024-08-05)
### 🐛 Bug Fixes
* restrict geopandas to below v1 until issue with `scheme` is fixed in next pygmt release ([`846056e`](https://github.com/mdtanker/polartoolkit/commit/846056ed9aa8de7105398d1014d91f5201141fce))
* remove nptyping as a dependency after issues with bool8 ([`6b97865`](https://github.com/mdtanker/polartoolkit/commit/6b97865002fed2469c32fea2276de352e90b416d))
### 🧰 Chores / Maintenance

## v0.5.0 (2024-08-04)
### 🐛 Bug Fixes
* add warning for deprecated origin shift parameters ([`728a218`](https://github.com/mdtanker/polartoolkit/commit/728a218aa3684f7990b3cc72abeb45d440e9c7bb))
* found bug in `utils.square_subplots` ([`f87f2e1`](https://github.com/mdtanker/polartoolkit/commit/f87f2e1ff96f2335c857cdb00b5594325469421e))
* drop band and spatial_ref variables from fetched grids ([`40f6ba2`](https://github.com/mdtanker/polartoolkit/commit/40f6ba2bb7e9fb7b9b4deda7ef895cc925bb4641))
### 📦️ Build
### 🧰 Chores / Maintenance
* add test deps to conda_install for testing conda releases ([`0cc2da2`](https://github.com/mdtanker/polartoolkit/commit/0cc2da230754d9f23138ca13d123b5798e93fa0c))
* updates from learn-scientific-python ([`07d79a9`](https://github.com/mdtanker/polartoolkit/commit/07d79a90213f8646bb909c6f5e03af99359b8106))
* update binder env wih polartoolkit v0.4.0 ([`47ff400`](https://github.com/mdtanker/polartoolkit/commit/47ff400f81c3c33f4ab626e94c5f526b8a22f832))
### ✏️ Formatting
* pre-commit fixes ([`463a225`](https://github.com/mdtanker/polartoolkit/commit/463a225babe3d84aa2d2589121eba632050aa386))
* auto style fix ([`65266e9`](https://github.com/mdtanker/polartoolkit/commit/65266e99e46481d88ffdf386899958a28417ed03))
* typo ([`a8e0342`](https://github.com/mdtanker/polartoolkit/commit/a8e03420f4af969933495fbc98abccfd1ef201b1))
### 📖 Documentation
* fix issues with autoapi and typehints ([`51f4266`](https://github.com/mdtanker/polartoolkit/commit/51f4266c8aee1d3a37021844c34a0e7441ebf442))
* rerun all doc notebooks ([`1c0fcb3`](https://github.com/mdtanker/polartoolkit/commit/1c0fcb33e5167a05a3bc9cbe986b81330ce62622))
* minor corrections to docs ([`64ee497`](https://github.com/mdtanker/polartoolkit/commit/64ee497779bf20a34793956fb59d510d4b9b9282))
* reduce notebook size with pygmt `dpi` parameter ([`46aadb8`](https://github.com/mdtanker/polartoolkit/commit/46aadb8c8727f620822811f2e0678ddc2007421d))
* add make setup instructions to contrib guide ([`18fe9cf`](https://github.com/mdtanker/polartoolkit/commit/18fe9cf9b76c964baea357e109533c4c492398ae))
* remove install info from readme ([`09ef175`](https://github.com/mdtanker/polartoolkit/commit/09ef175c3a9096ec014e3ee86880eb397fcd0f42))
* typos in notebook ([`d539795`](https://github.com/mdtanker/polartoolkit/commit/d539795396d9c8675e71c797492a17f3cd44e09b))
* fix missing instruction in contrib guide ([`8128a8f`](https://github.com/mdtanker/polartoolkit/commit/8128a8f0389f673dc91916f659b0e32417f7a3ea))
* move contrib guide into docs/ ([`3001032`](https://github.com/mdtanker/polartoolkit/commit/30010329bed35bd608b1f40f1c23bd403d639ad2))
* update license to 2024 for source files ([`127b173`](https://github.com/mdtanker/polartoolkit/commit/127b173745181ad5fc25686c3d873a4124147862))
* add citation info ([`d1281da`](https://github.com/mdtanker/polartoolkit/commit/d1281dab3b7c698ee7653be36fd8eaa560c2a861))
* update docs with numbered tutorials and other fixes ([`fdb7227`](https://github.com/mdtanker/polartoolkit/commit/fdb72271ac9b7cd7f5a59779a0ab6aa619992221))
* point binder link to tutorials ([`3fddb27`](https://github.com/mdtanker/polartoolkit/commit/3fddb27c5a1c28bf2d8b7807f1cf1913e8393b0f))
### 🚀 Features
* add 'easting' and 'northing' as defaults for plotting points ([`13b917b`](https://github.com/mdtanker/polartoolkit/commit/13b917b31e939f73fbe4a4764502bfa81b15eeaf))
* add earthaccess dependency and use for login credentials ([`1b87ca8`](https://github.com/mdtanker/polartoolkit/commit/1b87ca8f0b71bba73afd617cc06394b6a74b122e))
###  🎨 Refactor
* organize `plot_grd` and enable more mapping features for `basemap` ([`684e86e`](https://github.com/mdtanker/polartoolkit/commit/684e86e5957276f2e86fcd38fb1852c234a2a67f))
* remove kwargs for `add_gridlines` ([`e86144d`](https://github.com/mdtanker/polartoolkit/commit/e86144d707dedbddae98414eea3f0d27b6584cd9))
* change `origin_shift` parameter options

BREAKING CHANGE: please update your code for functions `plot_grd` and `basemap` to use the following options for `origin_shift`: 'x', 'y', 'both', 'initialize', or None. ([`4e751c8`](https://github.com/mdtanker/polartoolkit/commit/4e751c8d6fc848bc15cd04657d913af374af8e4e))
* mock import geopandas ([`ab3437d`](https://github.com/mdtanker/polartoolkit/commit/ab3437d17d856fe78e02f34596764e2937463004))
* use figshare sample shapefiles and remove data/ from repo ([`de684bb`](https://github.com/mdtanker/polartoolkit/commit/de684bb53dd0a09de969160ebf6c32e0f44628e2))
* remove quotes from scalebar ([`7995f9f`](https://github.com/mdtanker/polartoolkit/commit/7995f9fde71428e8e3f6cc58a917e26642a8ec8f))
### Other
*  ([`4424866`](https://github.com/mdtanker/polartoolkit/commit/4424866f96dd88f5948ad307d28c4b958ac40caf))

## v0.4.0 (2024-06-14)
### 🐛 Bug Fixes
* fix deprecated test measures boundaries ([`6f3cc23`](https://github.com/mdtanker/polartoolkit/commit/6f3cc232ac115bc0694fe0ba24506aad186d0e22))
* fix deprecated test alter region ([`eaad9c8`](https://github.com/mdtanker/polartoolkit/commit/eaad9c8a2f4136fbcaaab7ab193484d41a6c8484))
* fix fetch ice vel issue ([`02bf5e4`](https://github.com/mdtanker/polartoolkit/commit/02bf5e482cc1fd340da539897dff0d6d37972219))
* remove hash from mass_change ([`54cfe1b`](https://github.com/mdtanker/polartoolkit/commit/54cfe1b10a7edeea513920581668517a5c76d24b))
* fix tests for deprecations ([`890bc21`](https://github.com/mdtanker/polartoolkit/commit/890bc21be24465a64616c4e230fbdaddfba07172))
* remove alter region from get_regions ([`ab009d4`](https://github.com/mdtanker/polartoolkit/commit/ab009d48b13f8e270f160679456d088bb40b82f3))
* properly set default region in basemap ([`f5ca3da`](https://github.com/mdtanker/polartoolkit/commit/f5ca3daca7a025c61f492ef6f52e1cef473e4f6b))
* specify hemisphere in bedmap2 reference test ([`fd4e668`](https://github.com/mdtanker/polartoolkit/commit/fd4e668dac737510debb978e0613805a7b0d3a4d))
* for bedmachine grids, restore correct registration type and only resample after geoid added to grids ([`bf600b4`](https://github.com/mdtanker/polartoolkit/commit/bf600b4d445de63cd086edcad82c79fc206fde9c))
* remove fetch basement due to limited spatial extent of data ([`4cfc9e5`](https://github.com/mdtanker/polartoolkit/commit/4cfc9e511034349df2773833d3ef52875ef2e947))
* remove support for fetch ROSETTA grav and mag to align with goal of only providing common and widespread datasets ([`cc8d2af`](https://github.com/mdtanker/polartoolkit/commit/cc8d2af40947b2e8204e1ea3c16b3bc5f0a8e45b))
* update warning about grid region extract in plot_3d ([`b9be0ae`](https://github.com/mdtanker/polartoolkit/commit/b9be0ae43b8c38018c667b3998a5986ffd4426d4))
* update figure shifting of plot_3d ([`2ff91d3`](https://github.com/mdtanker/polartoolkit/commit/2ff91d345499d71dcc92786be7dd1860f6c60a0a))
* warning about histogram if grid is constant value ([`7d7647b`](https://github.com/mdtanker/polartoolkit/commit/7d7647be02bd076df3444b37dc35e2f37a8362a1))
* improve creation of colormap ([`25ea3b5`](https://github.com/mdtanker/polartoolkit/commit/25ea3b51ca9f00372e712ca37e9e2ec9bafdc7c1))
* correctly center interactive map ([`f0f5887`](https://github.com/mdtanker/polartoolkit/commit/f0f5887d6a5c7e3a0c8f8ebc8246c4f6930dbea1))
* change greenland region slightly ([`6953ce0`](https://github.com/mdtanker/polartoolkit/commit/6953ce03a0208f21c29548a61bcd6f46949f79fa))
* load deepbedmap instead of returning string ([`6840639`](https://github.com/mdtanker/polartoolkit/commit/684063917ac37b04eafdb2621deeeb7eff9d25cb))
* remove faulty layer name from fetch bedmachine ([`78a9879`](https://github.com/mdtanker/polartoolkit/commit/78a98799aeba844910a48bbe87b81f506f7e7932))
### 📦️ Build
* move deprecation from dev to normal dependencies ([`4a83f67`](https://github.com/mdtanker/polartoolkit/commit/4a83f6732f6c09938476a27803bbafbb94f2ef25))
* add deprecation package to dev deps ([`ac04b0d`](https://github.com/mdtanker/polartoolkit/commit/ac04b0d6f226629c191fee7dca2935c4d631005f))
* include environment.yml file as backup for issues with installing ([`8ef681f`](https://github.com/mdtanker/polartoolkit/commit/8ef681fe0935b55fc59e5f177222572996f5520b))
* set min pylint version ([`a93d83a`](https://github.com/mdtanker/polartoolkit/commit/a93d83a3ce35b6c4c56eee8b2edaf3ddafc63630))
* explicitly specify tqdm as dependency ([`287a4db`](https://github.com/mdtanker/polartoolkit/commit/287a4db768456da2a993f82559bd7b7f9c04ca9b))
### 🧰 Chores / Maintenance
* enable caching of test CI environment ([`f374dd2`](https://github.com/mdtanker/polartoolkit/commit/f374dd2c441627f6d3931d27c06e1c550310b468))
* switch CI from mini-conda to micromamba ([`69d182d`](https://github.com/mdtanker/polartoolkit/commit/69d182d66cb4e6a4f2d317b1b96a4dbd58cf9da3))
* remove pip local install in testing CI ([`563f98e`](https://github.com/mdtanker/polartoolkit/commit/563f98e140eaa22f43b1d62212a2f10ee84854bc))
* don't run fetch calls during test

This should drastically speed up testing, not overload GHA with downloading datasets, but means testing of fetch calls will need to be performed manual by deleting the local pooch cache and rerun all fetch calls / tests. ([`090f239`](https://github.com/mdtanker/polartoolkit/commit/090f239bbeed8718572595186ec2df9939360df9))
* add fetch mark to some tests ([`2f5771c`](https://github.com/mdtanker/polartoolkit/commit/2f5771c4f5c96079607c9e0a61d90f16387ed8c7))
* mark some tests as earthdata ([`bacb0fa`](https://github.com/mdtanker/polartoolkit/commit/bacb0faa436df41823e31b240115170343d18d3d))
* add codecov token ([`b05c119`](https://github.com/mdtanker/polartoolkit/commit/b05c1199b672cd4ca8639bcb00bb35cd797dc78d))
* add deprecation to old fetch measures boundaries function ([`262a0c2`](https://github.com/mdtanker/polartoolkit/commit/262a0c230a2b9b497f03378e5c5d716d9949d2a0))
* add deprecations to old fetch modis functions ([`7c5461e`](https://github.com/mdtanker/polartoolkit/commit/7c5461ec57252a4cb23310f40cd26c99002f1b8b))
* update fetch mass change test ([`9b2c282`](https://github.com/mdtanker/polartoolkit/commit/9b2c282138144649ed456d75d7f6abe6953d071f))
* exclude some md files from pre-commit ([`e4acbeb`](https://github.com/mdtanker/polartoolkit/commit/e4acbeb0a6ec47612f047e5fc62219aa6c10c333))
* update use of pylint ([`c886573`](https://github.com/mdtanker/polartoolkit/commit/c886573fadd0787cb3038921bc558d67319c9b91))
* update tests for bedmachine with reference change ([`c1200ca`](https://github.com/mdtanker/polartoolkit/commit/c1200ca6b2bae2946e23a4e9ffc6fc514febdb99))
* pre-commit autoupdate to monthly ([`1f977cc`](https://github.com/mdtanker/polartoolkit/commit/1f977cc504d19066d36d6811bdf5ca3a2a23c593))
* fix test ([`4ac5e0b`](https://github.com/mdtanker/polartoolkit/commit/4ac5e0b520aba72f2939fedb1d52045e707af036))
* misc fixes ([`699c02f`](https://github.com/mdtanker/polartoolkit/commit/699c02fd030340d7915cdaa3cb10a88a2cc0dccc))
* add greenland coast version to plotting functions ([`9dc125b`](https://github.com/mdtanker/polartoolkit/commit/9dc125bfcbfb75c97500557a0aeb94c8287eb01b))
* change arg `image` to `modis` ([`677b13e`](https://github.com/mdtanker/polartoolkit/commit/677b13e94b2c9789c156c698ae11a8e774f5ecb6))
* change default profile map cmap ([`c9396ac`](https://github.com/mdtanker/polartoolkit/commit/c9396ac3ef716395326c9401a7355fba2803154e))
* don't run tests on docs or style commits ([`8440e6f`](https://github.com/mdtanker/polartoolkit/commit/8440e6f754ef4ad8b716beeeb2d6ab5c4a64f61d))
* add tests for regions ([`77ea6f8`](https://github.com/mdtanker/polartoolkit/commit/77ea6f810f48be53497a273f41921e2b30551ab0))
* update ruff version in pre-commit ([`e13c0d9`](https://github.com/mdtanker/polartoolkit/commit/e13c0d948ff9597af8b3c890e84bbf67c53ea601))
* cleanup Make commands ([`eeab8f6`](https://github.com/mdtanker/polartoolkit/commit/eeab8f64e37f15cda48bf5897928192e2f016eb3))
* increase timeout for test shen moho ([`c769215`](https://github.com/mdtanker/polartoolkit/commit/c769215af61c3c4c0d12d53f5d9b5e042b18066a))
* remove pre-commit updates from changelog ([`611ff84`](https://github.com/mdtanker/polartoolkit/commit/611ff84c3ba5a5a31f6e4870f231dcd74c1f7053))
### ✏️ Formatting
* auto style fixes ([`3c3e155`](https://github.com/mdtanker/polartoolkit/commit/3c3e155af7fda31b6dc526cbfaa0c0ce9897e48f))
* style fixes ([`2022c6d`](https://github.com/mdtanker/polartoolkit/commit/2022c6d646bf5789c200c529060a4acd0e2ec9cf))
* fix cmap issues in plot_profile ([`4492445`](https://github.com/mdtanker/polartoolkit/commit/44924450860a496ea2a0f4e04f6962a87b317304))
* ignore pylint warnings ([`63e23a5`](https://github.com/mdtanker/polartoolkit/commit/63e23a5d3a9119ad6ca9882a1c09dd245a2fc286))
* auto fixes ([`52ac789`](https://github.com/mdtanker/polartoolkit/commit/52ac789133fcd98b0195a4815aa071c9a3918763))
* fixes for pylint ([`bf9c1cd`](https://github.com/mdtanker/polartoolkit/commit/bf9c1cde88e4c55323cdaa6fe60a66f9e7ac76c9))
* auto style fix ([`120e634`](https://github.com/mdtanker/polartoolkit/commit/120e634bb4dc66d93f4d6b0c61e0af089d710261))
* auto style fixes ([`b12c877`](https://github.com/mdtanker/polartoolkit/commit/b12c8777dcfff2881d94cd71b1e2fd56004b4236))
* add typing to function ([`51e25a9`](https://github.com/mdtanker/polartoolkit/commit/51e25a9491a0323b191c4781daf1b755a381b048))
* fix regions test styling ([`a8d9628`](https://github.com/mdtanker/polartoolkit/commit/a8d96288ebbd12d95156332b57c0fcf5c49d906d))
* reformat with new ruff version ([`363f8fa`](https://github.com/mdtanker/polartoolkit/commit/363f8fa3f1fc000ab62cd12fee9ede3a810c193b))
* auto style fixes from ruff/pylint ([`e1efbab`](https://github.com/mdtanker/polartoolkit/commit/e1efbab1eded88ebb42b256a09c4198e93f16842))
### 📖 Documentation
* typo and readme update ([`c0ffc0c`](https://github.com/mdtanker/polartoolkit/commit/c0ffc0ccb192abd267583430e4364da1ee5b18a2))
* spelling mistake ([`5fe90e5`](https://github.com/mdtanker/polartoolkit/commit/5fe90e513dddcff6ba67d9c013d365c2dd7be42a))
* add logo to README ([`b710923`](https://github.com/mdtanker/polartoolkit/commit/b710923dd541c29e34663a508f839f54534fa950))
* rerun all dataset gallery notebooks ([`35851b9`](https://github.com/mdtanker/polartoolkit/commit/35851b94ce28b64e14a572f507a7f4eea66c974b))
* major overhaul of docs structure to follow the Divio system ([`bbe3975`](https://github.com/mdtanker/polartoolkit/commit/bbe397560c83e5aeaa68b71a541c8890a42757a9))
* merge arctic and greenland dataset gallery ([`55abdf2`](https://github.com/mdtanker/polartoolkit/commit/55abdf22ac31e7da583c4da910eef3b9f3433fdc))
* add missing links ([`a8d0af5`](https://github.com/mdtanker/polartoolkit/commit/a8d0af508a5c2f603040e92c7c3db88135dc8f19))
* add instructions for EarthData login ([`18e5581`](https://github.com/mdtanker/polartoolkit/commit/18e5581ef6caecd78f8ba97efecd98651ba8f0b7))
* add release instructions to contributing guide ([`04d3ac2`](https://github.com/mdtanker/polartoolkit/commit/04d3ac23151f88fc468d8ff6223719187c4ce827))
* consolidate and update install instructions ([`399ba89`](https://github.com/mdtanker/polartoolkit/commit/399ba89dd8369ed82cfb44e4332e339d66b0a8f8))
* rerun notebooks ([`c5fe426`](https://github.com/mdtanker/polartoolkit/commit/c5fe426dc7bc9d6bf195bca5a48dda14634757ba))
* clarify all mentions of format of region tuples ([`1e8c09f`](https://github.com/mdtanker/polartoolkit/commit/1e8c09f78d3e40fef526d3a015b437ee456daf8f))
* clarify project scope and goals ([`11ef365`](https://github.com/mdtanker/polartoolkit/commit/11ef365e1ecd3d4c548354021d422fe2935b0d6a))
* add rosetta grav/mag to gallery datasets ([`690e747`](https://github.com/mdtanker/polartoolkit/commit/690e74730970669bae2c4305782f3911aa0ac47a))
* add measures ice boundaries to dataset gallery ([`60db1c9`](https://github.com/mdtanker/polartoolkit/commit/60db1c9e211fed966593cdd3449ab4c9768b6fb9))
* add basal melt rate to dataset gallery ([`02ee6f7`](https://github.com/mdtanker/polartoolkit/commit/02ee6f74b224b6032da96cb80624ba5507b120aa))
* add geomap geology to dataset gallery ([`4277b7c`](https://github.com/mdtanker/polartoolkit/commit/4277b7cf635c0354d8ebcdac7d23a38cfa5cd0df))
* add groundline to dataset gallery ([`39b142e`](https://github.com/mdtanker/polartoolkit/commit/39b142eea3854e57602bf2ac45d6517cf4275cfb))
* add mass change to dataset gallery ([`0476df2`](https://github.com/mdtanker/polartoolkit/commit/0476df2269a31fc3b6a5e27e37909cdce3523ed9))
* add magnetics to dataset gallery ([`f0cadd9`](https://github.com/mdtanker/polartoolkit/commit/f0cadd994a47e484111e9f64866400c63ffb6e6d))
* add ice velocity to dataset gallery ([`3e8a022`](https://github.com/mdtanker/polartoolkit/commit/3e8a022288f8d68a95993412489090ec4874de92))
* add gravity to dataset gallery ([`80d1a6c`](https://github.com/mdtanker/polartoolkit/commit/80d1a6cac00dbc6890350bf4f7298398de62ad66))
* add GIA to dataset gallery ([`5e2eced`](https://github.com/mdtanker/polartoolkit/commit/5e2eced0dfcc3cef7d38f96ed2e40e93623ba217))
* use full citations in dataset gallery ([`df634ba`](https://github.com/mdtanker/polartoolkit/commit/df634ba8424661f04bc254795df03dc3ecedde7a))
* add GHF to dataset gallery ([`95826c1`](https://github.com/mdtanker/polartoolkit/commit/95826c1a043bd1f6d944f256c856615cd15062df))
* rerun notebook ([`2c44c08`](https://github.com/mdtanker/polartoolkit/commit/2c44c08ee53ed3e9622a3a4090162ef092254847))
* add modis mog to index ([`2a041a8`](https://github.com/mdtanker/polartoolkit/commit/2a041a8bacc1d48e51a473896f946e1db0b6716c))
* rerun all docs ([`9984c8d`](https://github.com/mdtanker/polartoolkit/commit/9984c8dc1aea5f33a4c789f0fd55ca9506b79052))
* add more dataset notebooks ([`4526aab`](https://github.com/mdtanker/polartoolkit/commit/4526aab16d0f446084d698b660075072811b443f))
* edit some docstrings ([`66a1ef5`](https://github.com/mdtanker/polartoolkit/commit/66a1ef5201273f3ecd139c2c2830cc073bc86c82))
* reorganize available dataset gallery ([`99ac4a2`](https://github.com/mdtanker/polartoolkit/commit/99ac4a2699db63721928e82887a20c8a4e38b26a))
* add more dataset notebooks ([`41908f0`](https://github.com/mdtanker/polartoolkit/commit/41908f0261eea6fc6c193d9723002af332615b99))
* update references ([`520823f`](https://github.com/mdtanker/polartoolkit/commit/520823fb3ed9904aba0eccf74ec35796287e109c))
* mention dataset gallery in fetch walkthrough ([`d6387b3`](https://github.com/mdtanker/polartoolkit/commit/d6387b35ddfdc13fda64735067b7361b84bfc4ee))
* add bedmachine dataset notebook ([`2651e46`](https://github.com/mdtanker/polartoolkit/commit/2651e4651f3f43128485662cbaf0fa283e5309e3))
* update bedmap2 dataset notebook ([`1dee9e9`](https://github.com/mdtanker/polartoolkit/commit/1dee9e9a548aac0f98109397ea12d9b837548bfc))
* update dataset gallery template ([`1d100c3`](https://github.com/mdtanker/polartoolkit/commit/1d100c39ec1fe7447a7902b95949b3f9437b15a9))
* add available dataset gallery ([`7932719`](https://github.com/mdtanker/polartoolkit/commit/79327193a98f95e20aa5035547b901e04c48b339))
* conform fetch walkthrough to standard figure creation style ([`a740223`](https://github.com/mdtanker/polartoolkit/commit/a740223334729e440a9fdf7e0e0c042905f7c47e))
* switch binder env to separate repo, remove launch buttons ([`921f091`](https://github.com/mdtanker/polartoolkit/commit/921f091692ec2f4e601abce225f35e31351f0be5))
* try nbgitpull link for launch binder buttons ([`5ca4de9`](https://github.com/mdtanker/polartoolkit/commit/5ca4de9d4dc38c391d787df092ea5fa897c27ef3))
* fix issues with subplot_layout example ([`9e46dbd`](https://github.com/mdtanker/polartoolkit/commit/9e46dbd65fa2b84a4b4ca1a910ebb768392dd2f1))
* remove python version and nodefaults from binder env ([`9c46665`](https://github.com/mdtanker/polartoolkit/commit/9c46665e69341c9be7e466f78bfc22b961b14a1c))
* simplify binder env ([`8e2c8ed`](https://github.com/mdtanker/polartoolkit/commit/8e2c8ed2a8941689abd49031f57f8a06811a4e9c))
* update contributing guide ([`777b662`](https://github.com/mdtanker/polartoolkit/commit/777b662b6c88b6c3f154969b0e35e0c69f62253e))
* reuse readme material in RTD index ([`8791e96`](https://github.com/mdtanker/polartoolkit/commit/8791e9604e007417d4e7240a37fdf7d17f98000c))
### 🚀 Features
* add points option to basemap ([`474cde7`](https://github.com/mdtanker/polartoolkit/commit/474cde7a0784d2f3240ec1fe0c6629f47e18bbdd))
* add region_ll_to_xy function ([`31b054c`](https://github.com/mdtanker/polartoolkit/commit/31b054c2701fc586eb6812c2a123933564eb1003))
* add ice velocity for Greenland ([`9ad6fe7`](https://github.com/mdtanker/polartoolkit/commit/9ad6fe769a56dba0d5468cf45a974656f36b2c01))
* add etopo data for the arctic/greenland ([`2910e30`](https://github.com/mdtanker/polartoolkit/commit/2910e30cf60d42a1a2f0acaed7f8a801814d32cb))
* add mass change for Greenland ([`56bd01a`](https://github.com/mdtanker/polartoolkit/commit/56bd01ac07b02c9fae3f205dc80ddb8a2ef74c03))
* enable greenland support for profiles ([`548f626`](https://github.com/mdtanker/polartoolkit/commit/548f626422eda409c653faee66359d9b3e64f4ea))
* add function to extract default hemisphere ([`b8b4f0a`](https://github.com/mdtanker/polartoolkit/commit/b8b4f0acb1ae3354d1aef356312a7832a3a34650))
* add geoid for Greenland and change geoid region for Antarctica ([`4987e0b`](https://github.com/mdtanker/polartoolkit/commit/4987e0b7b5e1e76f13754aafb2ff212ef07d8d55))
* add bedmachine data from Greenland ([`cfa5fff`](https://github.com/mdtanker/polartoolkit/commit/cfa5fff2dd5c70072a731e067c7c8a729897a914))
* add cbar_labels kwarg to plot_3d ([`9d26d06`](https://github.com/mdtanker/polartoolkit/commit/9d26d06d7adc700928f37fc18388fe9397053ee9))
* add cbar_perspective arg to plot_3d ([`25f1bf7`](https://github.com/mdtanker/polartoolkit/commit/25f1bf70530aa548a17433d852a0f3d162276f3b))
* enable passing single grid to plot_3d ([`b71001a`](https://github.com/mdtanker/polartoolkit/commit/b71001a5c4ebf36c32729a62caf4f4332480dbfc))
* add function for combined min max of several grids ([`6d3c294`](https://github.com/mdtanker/polartoolkit/commit/6d3c294ab9e66bb0db192175547832c638ad751b))
* allow passing .cpt file to colorbar histogram ([`e633789`](https://github.com/mdtanker/polartoolkit/commit/e6337894f6f8aba39b1e01ef6a1e3e6735388972))
* add inset map option for greenland ([`5d36db4`](https://github.com/mdtanker/polartoolkit/commit/5d36db441d8709c7439a2730727a0b92e96255c5))
* add hemisphere arg to most plotting functions

BREAKING CHANGE: this alters a vast majority of the code! ([`05cb358`](https://github.com/mdtanker/polartoolkit/commit/05cb3583482967deaa75abb9f1bd5e91b7d5ee96))
* add subset grid function ([`ab2e60a`](https://github.com/mdtanker/polartoolkit/commit/ab2e60a81e70e24e6347c014d4a8a51bb2b078eb))
* add BAS greenland groundingline ([`d29863e`](https://github.com/mdtanker/polartoolkit/commit/d29863ef8f3ae0f212de571f9b96b2509324292a))
* add pen kwarg to show_region ([`308eac0`](https://github.com/mdtanker/polartoolkit/commit/308eac08cb0bd1528f24432bb6c6f8ff9ca1f601))
* add fetch MODIS Greenland ([`3e78b8e`](https://github.com/mdtanker/polartoolkit/commit/3e78b8e79bd2de8ed8ebf8e4fe23391da0bc5fe1))
* add some Greenland regions ([`29fa593`](https://github.com/mdtanker/polartoolkit/commit/29fa593937f90f131c830d0185460b9b586ec6ff))
* add Arctic/Greenland to utils/maps ([`503b6a1`](https://github.com/mdtanker/polartoolkit/commit/503b6a11fcf475410002ce9fc73473c2c28324d9))
###  🎨 Refactor
* change module name from `profile` to `profiles` to match plural style of other modules.

BREAKING CHANGE: please update all import statements to use `profiles` instead of `profile`! ([`84524e3`](https://github.com/mdtanker/polartoolkit/commit/84524e3b2df06156cbdfbbd13b5eb7a17cff1965))
* update deprecated pandas delim_whitespace ([`f70476a`](https://github.com/mdtanker/polartoolkit/commit/f70476a01a916e7b2ff59ff0a4c661902c59408f))
* move alter_region from utils to regions ([`f95ced0`](https://github.com/mdtanker/polartoolkit/commit/f95ced0c1684bb53e1a3aaa351bae5e9a3bb46e6))
* combine fetch `modis_moa` and `modis_mog` to `modis`

BREAKING CHANGE: make sure to update your code to use the new function `fetch.modis()` and specify MoG vs MoA with parameter hemisphere = "south" or "north" ([`89e50d2`](https://github.com/mdtanker/polartoolkit/commit/89e50d2a65ae5a5b26092f09a0c850718b69e458))
* add hemisphere and remove kwargs from call to geoid ([`db5241e`](https://github.com/mdtanker/polartoolkit/commit/db5241e5c0cba7a741557700ad8f8ace59b853e7))
* remove unnecessary verbose and kwargs for some fetches ([`fdab8fe`](https://github.com/mdtanker/polartoolkit/commit/fdab8fe9e2bdc5684f755f2782e590109e3180b0))
* log info instead of error in mask_from_shp ([`fd2e768`](https://github.com/mdtanker/polartoolkit/commit/fd2e7680d8ceaed1244bc1e8de4fa7989a73862f))
* use environment variable for setting hemisphere in profile ([`3be6d86`](https://github.com/mdtanker/polartoolkit/commit/3be6d86250c853ca08e79998027f7f279e1b8cba))
* if available, use environmental variables for setting hemisphere throughout code

To utilize this feature, either set a system environment variable POLARTOOLKIT_HEMISPHERE equal to either south or north, or in your python file/notebook use os.environ to set this value for the current session. ([`d756838`](https://github.com/mdtanker/polartoolkit/commit/d756838994ac7c8eb98b6803b215ad7e08a0ad3e))
* rename fetch `measures_boundaries` to `antarctic_boundaries`

BREAKING CHANGE: make sure to update your code with the new function name! ([`6951fe2`](https://github.com/mdtanker/polartoolkit/commit/6951fe201c1ad7d941a17dddaeb3b48912797d80))
* use new set_cmap function in plot_3d ([`1e68076`](https://github.com/mdtanker/polartoolkit/commit/1e6807623e5a91d68607c43d35945c854e6fbc2f))
* set cbar position in plot_3d ([`ecf53cf`](https://github.com/mdtanker/polartoolkit/commit/ecf53cf2b0ecdaa1b2b78830aed80ed17e661b95))
* changed default shading to false in plot_3d ([`7e4feed`](https://github.com/mdtanker/polartoolkit/commit/7e4feed880dc44fd2a86563b5663feb54ee50281))
* remove default vlims for plot_3d ([`3f59d10`](https://github.com/mdtanker/polartoolkit/commit/3f59d105c8751bff5f498c1281ca05f63f57bbda))
* change default point  color in plot_profile ([`e895aed`](https://github.com/mdtanker/polartoolkit/commit/e895aed8d501b8d775e9efb4ae5ac2b246fcf45b))
* update np random number generator ([`0daaffc`](https://github.com/mdtanker/polartoolkit/commit/0daaffc99c0650971a3d00ce17d503d58ce05ce3))
### Other
*  ([`49e95cb`](https://github.com/mdtanker/polartoolkit/commit/49e95cb854a9938095cf277d872db148a84fce5c))
*  ([`cb7ee9d`](https://github.com/mdtanker/polartoolkit/commit/cb7ee9d64117a96e0220b9d2ff3f90c1f1b71f7d))

## v0.3.3 (2024-04-23)
### 🐛 Bug Fixes
* pinning issue in workflow ([`1578894`](https://github.com/mdtanker/polartoolkit/commit/1578894981c4c33ea00798ea163a620bc5128d28))
### 📦️ Build
* make semantic release GHA need changelog success ([`adf7edb`](https://github.com/mdtanker/polartoolkit/commit/adf7edb40d32a4e9090b9cfaff7c0a51f5299424))
### 🧰 Chores / Maintenance
* pin python semantic release version ([`73a0fbd`](https://github.com/mdtanker/polartoolkit/commit/73a0fbd00bafa012ca0fb63518d3e07a4580f24f))

## v0.3.2 (2024-04-23)
### 🐛 Bug Fixes
* reduce accuracy check for test resampling grids ([`93fca6f`](https://github.com/mdtanker/polartoolkit/commit/93fca6f44b69ae46672fcd74169d2c86d725a062))
### 📦️ Build
### 🧰 Chores / Maintenance
* fix semantic release action ([`6b12799`](https://github.com/mdtanker/polartoolkit/commit/6b12799d984917e9e4693bf8003c6869fc7f350d))

## v0.3.1 (2024-02-22)
### 🐛 Bug Fixes
* increase timeout for shen-2018 moho fetch ([`8d9e183`](https://github.com/mdtanker/polartoolkit/commit/8d9e183de4aca1cc985a4824414b3107c50905e6))
* delete tmp.nc file in fetch call ([`ee16bdc`](https://github.com/mdtanker/polartoolkit/commit/ee16bdc10f000d1d05cabc889890548fd2f72dbe))
### 📦️ Build
* extend support to python 3.12 ([`f68def9`](https://github.com/mdtanker/polartoolkit/commit/f68def9b1156f020d4e222bf27279293eddda51f))
### 🧰 Chores / Maintenance
* remove shen-2018 hash ([`fbefe40`](https://github.com/mdtanker/polartoolkit/commit/fbefe4031c2b241fdde5b6375bc16b504ec6b9ac))
* fix test ghf ([`c7c66f6`](https://github.com/mdtanker/polartoolkit/commit/c7c66f60fce00b50524abc6eca01238f40b932e1))
* update changelog template ([`e4bd1d9`](https://github.com/mdtanker/polartoolkit/commit/e4bd1d923fb635ac2c9724a9c8babbe3cd9a4660))
* reduce sig figs for test_ghf ([`393aadc`](https://github.com/mdtanker/polartoolkit/commit/393aadc4394bc0da73d4eb38e1318bc04f374636))
* list packages after installing local for test GHA ([`51250b4`](https://github.com/mdtanker/polartoolkit/commit/51250b4dfa2a8c59380fe16d5f00e68edf453e2d))
* update testing_env and add make command ([`4ee6945`](https://github.com/mdtanker/polartoolkit/commit/4ee6945f06c39447a6447f3563898c867696fb5c))
* add hashes to all fetch calls ([`91eb9a8`](https://github.com/mdtanker/polartoolkit/commit/91eb9a83fcdd247b7a55491b0654ae5734c46fed))
* move contributing file ([`05dff10`](https://github.com/mdtanker/polartoolkit/commit/05dff10fb72fa82a4e24ecf6e47a73d40cdc9c63))
* add test_fetch make command ([`2a0adb3`](https://github.com/mdtanker/polartoolkit/commit/2a0adb30cdc669b602d6b0f5cdc402b3f5306872))
* add ignores to pre-commit ([`3fffd15`](https://github.com/mdtanker/polartoolkit/commit/3fffd15defb6de572e2f8caa1e0380a760ac74dd))
* fix changelog template spacing ([`cabc14a`](https://github.com/mdtanker/polartoolkit/commit/cabc14a644d10bf7eafe26c223a5542e17f10bf4))
### 📖 Documentation
* update install instructions ([`70faf04`](https://github.com/mdtanker/polartoolkit/commit/70faf0485d1157dda42e3f40ca462d89d87fafad))
* fix links to contrib guide ([`5c6af8d`](https://github.com/mdtanker/polartoolkit/commit/5c6af8de8ab984fa645f868ba23eadd492f0d4b7))

## v0.3.0 (2024-02-18)
### 📦️ Build
* add requests to deps ([`15ea6e8`](https://github.com/mdtanker/polartoolkit/commit/15ea6e838a980f7c7383819122013fc103b9005f))
* switch GHA from hynek to build and twine ([`91c6ca9`](https://github.com/mdtanker/polartoolkit/commit/91c6ca9d11afd8197bf3fa043c7c8d2a5e9617d3))
* explicitly include packages ([`a76a0e9`](https://github.com/mdtanker/polartoolkit/commit/a76a0e99c2aaec43b0a35218136609a970450b64))
### 🧰 Chores / Maintenance
* ignore auto update changelog commits in changelog ([`01a092e`](https://github.com/mdtanker/polartoolkit/commit/01a092ee4cc77a60270deb2477de6cec70b002cd))
* mark test mass change as issue ([`a3bde43`](https://github.com/mdtanker/polartoolkit/commit/a3bde4376c336d82b1e600fdf7306e7cf580267b))
* add issue marker to imagery and basal melt fetches ([`95bce69`](https://github.com/mdtanker/polartoolkit/commit/95bce69504dec80d3581b6ed864c9aaf892f70a9))
* make fig an arg of plot_grd ([`e762a5a`](https://github.com/mdtanker/polartoolkit/commit/e762a5a7518932d5f630d7087517912055f3a0e3))
* add kwarg arg to add_scalebar ([`df374ba`](https://github.com/mdtanker/polartoolkit/commit/df374ba838dea59c454cca2314a0b93841d1370c))
* specify args in functions ([`9febc0e`](https://github.com/mdtanker/polartoolkit/commit/9febc0e46b3d405c7c83a12653344a59c3fb629b))
* remove dependa-bot commits from changelog ([`44a322c`](https://github.com/mdtanker/polartoolkit/commit/44a322c4059a69e9af735232799d6583f4b86e84))
* fix _static path warning ([`257f6e7`](https://github.com/mdtanker/polartoolkit/commit/257f6e728e6fd24d695b3b8ee5acc3140b260aa7))
* add skip ci  to changelog commit message ([`6b665ae`](https://github.com/mdtanker/polartoolkit/commit/6b665aed5085b9f9166c996b2bf1cea91c8a5661))
* update changelog an main pushes ([`2dca0fc`](https://github.com/mdtanker/polartoolkit/commit/2dca0fc619b303c9b73c6c69ee15d14165e6be51))
* editing workflows ([`a059f50`](https://github.com/mdtanker/polartoolkit/commit/a059f50061eb8e4399e2aebed5b1491f44dd0e89))
* collect usage stats ([`b5d16ac`](https://github.com/mdtanker/polartoolkit/commit/b5d16ac3792e3ef7791138274d1c069a108cc110))
### ✏️ Formatting
* typos and type issues ([`5c478fb`](https://github.com/mdtanker/polartoolkit/commit/5c478fb124f93e3b1ff09ccd87dc1fa5af7ede74))
* fixes and ignore .bib ([`c154f89`](https://github.com/mdtanker/polartoolkit/commit/c154f89b8ba97a3ce3359fa5d1fc266e51e8dcdd))
* formatting ([`0b294a1`](https://github.com/mdtanker/polartoolkit/commit/0b294a18d71d909bcbbd398dc0634b548faf7165))
* fix style errors ([`639ba42`](https://github.com/mdtanker/polartoolkit/commit/639ba42c996db46125b5889e5ae35e09217296d7))
* fix indent ([`efee6d7`](https://github.com/mdtanker/polartoolkit/commit/efee6d702e7db92b5f32d4136be2b0b18eeaf54a))
### 📖 Documentation
* update RTD links to polartoolkit ([`e5376b3`](https://github.com/mdtanker/polartoolkit/commit/e5376b359e0b6bbadaaadb02cd8731be77033517))
* fix disclaimer ([`8075024`](https://github.com/mdtanker/polartoolkit/commit/80750247c570b8d98774b02dcab64e33fd57d56c))
* re-run all docs ([`0776879`](https://github.com/mdtanker/polartoolkit/commit/07768799052e77a59e36783bc2350e352d3ad4a7))
* add pygmt and pip to RTD.env ([`9a8ae51`](https://github.com/mdtanker/polartoolkit/commit/9a8ae51b6cc51d9b798252e67932a05ec1148742))
* back to conda for RTD ([`d2a4c1f`](https://github.com/mdtanker/polartoolkit/commit/d2a4c1f7e8d610fefd358917d61459d6b38939c0))
* switch RTD from mamba to python ([`aeab560`](https://github.com/mdtanker/polartoolkit/commit/aeab560cc1872299c9be10e27f5f8f5327349091))
* switch RTD from conda to pip ([`18cf7fb`](https://github.com/mdtanker/polartoolkit/commit/18cf7fb89e69a28be8ea10818bba28b2c92ccd3f))
* update RTD env and instructions ([`f142c6d`](https://github.com/mdtanker/polartoolkit/commit/f142c6d5a1061321ad1d279078fc0a8c7155604e))
* remove JOSS folder ([`dc0a421`](https://github.com/mdtanker/polartoolkit/commit/dc0a421e09ac8ad1926a7dea2ba078fc6e1e8e17))
* add .bib and enable sphinxcontrib-bibtex ([`bf90dc0`](https://github.com/mdtanker/polartoolkit/commit/bf90dc0456a2284300fea0b0dd8f74fe361d913b))
* add paper template for JOSS ([`22feca3`](https://github.com/mdtanker/polartoolkit/commit/22feca31678a6b0e68cf893bfe9d008b48649d04))
### 🚀 Features
* add plotting geomap faults function ([`3e87e60`](https://github.com/mdtanker/polartoolkit/commit/3e87e605b0dfa3805d94bbb6b3a20bca18c23a55))
* add cmap arg to points of plot_grd ([`f4fc948`](https://github.com/mdtanker/polartoolkit/commit/f4fc94832c110df855705d2f93a186a41b7c8ee2))
* add basemap imagery option to plot_grd ([`b7ad2ee`](https://github.com/mdtanker/polartoolkit/commit/b7ad2eecb08616f71126d1fc690b4315127922bc))
* make default scalebar have white box ([`3c879d5`](https://github.com/mdtanker/polartoolkit/commit/3c879d5f477e94dfb9986f2c334da8433fd15036))
* add water thickness layer to bedmap2 fetch ([`8f26e46`](https://github.com/mdtanker/polartoolkit/commit/8f26e461401e088ff707482151f4ba8e98696559))
* add / update pre-defined regions ([`4ce3d8d`](https://github.com/mdtanker/polartoolkit/commit/4ce3d8d1b35ba8716a4e74713d20de86d502b45a))

## v0.2.1 (2024-01-29)
### 🐛 Bug Fixes
* add "+ue" unit to regions in meters for lat long projections ([`fa67b53`](https://github.com/mdtanker/polartoolkit/commit/fa67b5367a94f362e040c210c547202d05976922))
* fixes lines between layers in cross-sections ([`7eaaf64`](https://github.com/mdtanker/polartoolkit/commit/7eaaf64629847a168d4249096b18e336e3c5a5a2))
* fix pandas copy warning ([`48ce7a7`](https://github.com/mdtanker/polartoolkit/commit/48ce7a7bf109868ddf23f04ba55d23898e2246e2))
### 📦️ Build
* add antarctic_plots package for import warning ([`bb4c134`](https://github.com/mdtanker/polartoolkit/commit/bb4c134b6e78377d426781ef1f601a5ba171000b))
* add lower version limit to PyGMT ([`4f3a837`](https://github.com/mdtanker/polartoolkit/commit/4f3a837de38852697ee85ffb341aadeaa9bb8c9b))
### ✏️ Formatting
* line too long ([`51e0143`](https://github.com/mdtanker/polartoolkit/commit/51e0143b571a8e8d901f9adaab6e3843fed8e823))
* spelling and formatting ([`a7347ba`](https://github.com/mdtanker/polartoolkit/commit/a7347ba6ca94185414ac3d0a0eec1bb1ef095bfb))
* spelling  mistakes ([`4c29294`](https://github.com/mdtanker/polartoolkit/commit/4c2929402b66903b2ddc0688333b932c44bba978))
*  ignore binder env file in pre-commit ([`9aa1c30`](https://github.com/mdtanker/polartoolkit/commit/9aa1c309352965131c40b3d88c4e16afbcdf48a6))
### 📖 Documentation
* clicking on logo directs to homepage ([`841e9e5`](https://github.com/mdtanker/polartoolkit/commit/841e9e5f38dcfce0fb847bebb7bcad050ce87069))
* rerun notebooks ([`eb5e47d`](https://github.com/mdtanker/polartoolkit/commit/eb5e47d53ee793688b461b374c6ee1e32bc00d82))
* update binder env, links, and codecov link ([`4854853`](https://github.com/mdtanker/polartoolkit/commit/4854853a2072632dbdbd3000f657cf717b7f6d15))

## v0.2.0 (2024-01-26)
### 📦️ Build
### 🧰 Chores / Maintenance
* fixes ([`d1cc66a`](https://github.com/mdtanker/polartoolkit/commit/d1cc66a98a2653db0b57593cabec9ca6da2e3878))
* formatting ([`f9ffa54`](https://github.com/mdtanker/polartoolkit/commit/f9ffa54b599ab1ab38af8d267a3daf1c98ae8bc0))
* formatting ([`52fd5bb`](https://github.com/mdtanker/polartoolkit/commit/52fd5bb3cf08927f5211ae7b4f2a14dacc310dbf))
* rename module ([`70f7d18`](https://github.com/mdtanker/polartoolkit/commit/70f7d18b7973ededb3b90ca809774160f2c8a1b4))
* formatting ([`6e30c4c`](https://github.com/mdtanker/polartoolkit/commit/6e30c4ca30a44ea7cd1353f5703fd4414816ae7e))
* switch from antarctic-plots to polartoolkit ([`bac23a9`](https://github.com/mdtanker/polartoolkit/commit/bac23a9a4c0c5e7059c42edc0892c96174dcc0dc))
### ✏️ Formatting
* formatting ([`d15fe92`](https://github.com/mdtanker/polartoolkit/commit/d15fe924d6097c64b410e4a0802b63774e17074e))
### 📖 Documentation
* update descriptions to not only focus on Antarctica ([`b08a509`](https://github.com/mdtanker/polartoolkit/commit/b08a509cf25956586282e905a3786d081461ae6e))
* add favicon and dark and light logos ([`77f9835`](https://github.com/mdtanker/polartoolkit/commit/77f9835fc76d9415b0ace6d93829560900d8eb72))
* add logo to docs ([`c26f850`](https://github.com/mdtanker/polartoolkit/commit/c26f8502ced995515b4ee40440c062b67b329b6b))
* fix changelog template ([`31083e3`](https://github.com/mdtanker/polartoolkit/commit/31083e3d729cd4fb17f2149665610c5ba7fcd2c0))
###  🎨 Refactor
* force a major version bump.

BREAKING CHANGE: ([`cc2ecda`](https://github.com/mdtanker/polartoolkit/commit/cc2ecdaae48b4300ae6d41485076273fa612ce64))

## v0.1.0 (2023-12-10)
### 🐛 Bug Fixes
* change default layer names in profile ([`801611f`](https://github.com/mdtanker/polartoolkit/commit/801611f196c1b10a0f95c8829dbcf0370c890ccf))
* avoid resampling default layers for profile ([`1f75666`](https://github.com/mdtanker/polartoolkit/commit/1f75666f7f7a18393a2bc9839e3d70344d5f0fda))
* warning for resampling default layers ([`543fe60`](https://github.com/mdtanker/polartoolkit/commit/543fe60a9e74c1282230813a7ccaef5e7d9fbef7))
* issue with plot_grd cpt_lims ([`548da1d`](https://github.com/mdtanker/polartoolkit/commit/548da1d66354b739f39deb7911c51b80d18d762f))
* fixing self imports ([`3e806df`](https://github.com/mdtanker/polartoolkit/commit/3e806df8840ce4653a3f438b87140bd26afab37b))
* switch module import style ([`c61552a`](https://github.com/mdtanker/polartoolkit/commit/c61552a7aadf01bdb15611518ae98ca77ad06a50))
* fixing typing cast ([`0405ad3`](https://github.com/mdtanker/polartoolkit/commit/0405ad3e68ebcbf01b4beec134dd47a6e27d530a))
* specify kwargs to shorten function ([`acaf8d9`](https://github.com/mdtanker/polartoolkit/commit/acaf8d98a8f8c617d0c028393ac1b0a27b50c45c))
* increase default colorbar font size ([`fc86e93`](https://github.com/mdtanker/polartoolkit/commit/fc86e93bd194b997ab41bca02834f3029e94aaf7))
* various fixes to fetch.py

leftover fixes after refactoring code to pass new formatting / linting / type checking procedures ([`4da7fc1`](https://github.com/mdtanker/polartoolkit/commit/4da7fc11f2630240a8f439b05cd060321e6e81f3))
* various fixes to maps.py

leftover fixes after refactoring code to pass new formatting / linting / type checking procedures ([`6b7b25c`](https://github.com/mdtanker/polartoolkit/commit/6b7b25c85b82391dcc2e7da221008cc572512aca))
* various fixes to utils.py

leftover fixes after refactoring code to pass new formatting / linting / type checking procedures ([`02d105d`](https://github.com/mdtanker/polartoolkit/commit/02d105d19a0c33e4c78787795ef7b926decd4ddc))
* various fixes to profile.py

leftover fixes after refactoring code to pass new formatting / linting / type checking procedures ([`aa43a85`](https://github.com/mdtanker/polartoolkit/commit/aa43a85c3cae6ee7d3b3ee89a93ba305869a7715))
* various fixes to regions.py

leftover fixes after refactoring code to pass new formatting / linting / type checking procedures ([`4f732aa`](https://github.com/mdtanker/polartoolkit/commit/4f732aa2385ddefd072b977edd63b63c5487ddea))
* change default layer pen to black ([`b2c1e74`](https://github.com/mdtanker/polartoolkit/commit/b2c1e74f1d32b6b06aedbcd69eb3e2a5bf6ec00a))
* change default inset region box to red ([`627b2bd`](https://github.com/mdtanker/polartoolkit/commit/627b2bd5aabe9c18edbd2ed65c6059b04d6c695c))
### 📦️ Build
* fix path to docs ([`4bbbd96`](https://github.com/mdtanker/polartoolkit/commit/4bbbd9639a75a50305e28c8102afdda58b774534))
* add ipython to interactive deps ([`4c706ad`](https://github.com/mdtanker/polartoolkit/commit/4c706ad43cbeb2fb5547669a65115ba911f83061))
* remove isort required import ([`a82ea7a`](https://github.com/mdtanker/polartoolkit/commit/a82ea7a8da80abd57825f7b3fa50a9fefac23a3b))
* add jupyterlab to dev deps ([`b370b6f`](https://github.com/mdtanker/polartoolkit/commit/b370b6f5856a1be5d4feb894a10d0a0fa3aac615))
* configure pytest ([`6c7f351`](https://github.com/mdtanker/polartoolkit/commit/6c7f351fa186267ff7f0cd709042d890cbcdb06f))
* update pyproject.toml deps and info ([`5653436`](https://github.com/mdtanker/polartoolkit/commit/5653436e601f5babed7deb2de265ad0fcab5678e))
* update env folder ([`9a4fa68`](https://github.com/mdtanker/polartoolkit/commit/9a4fa685d000db6969af2c15e64d4ca2106bd0de))
* update github actions ([`a1c5644`](https://github.com/mdtanker/polartoolkit/commit/a1c5644c22c40f8425363eb6026204da7da46e9b))
### 🧰 Chores / Maintenance
* reduce sig figs of fetch tests ([`15e5c3d`](https://github.com/mdtanker/polartoolkit/commit/15e5c3d7b09e4df2b68fd96637e4b0bb332faef3))
* match test env to pyproject ([`155f1cf`](https://github.com/mdtanker/polartoolkit/commit/155f1cf44fe9917a2806175b57d0be7d7a2aee61))
* switch python 3.8 to 3.9 for CI testing ([`1a17424`](https://github.com/mdtanker/polartoolkit/commit/1a17424fa5bcb3d6ad542ecb8c23ec8c5c7fede3))
* exclude some tests for CI ([`ac535d9`](https://github.com/mdtanker/polartoolkit/commit/ac535d93cb90c1c7a7f733b7b5bf9aa24ab097d8))
* update binder env and links ([`fe80114`](https://github.com/mdtanker/polartoolkit/commit/fe801140093564356fff7c3d535c4eb2d53bbc4e))
* type checking fixes ([`6ec45e1`](https://github.com/mdtanker/polartoolkit/commit/6ec45e1629314dc47c42bfbbed0ba1852b721600))
* update changelog template ([`1418eb8`](https://github.com/mdtanker/polartoolkit/commit/1418eb877aecc666e9427954fa58dd0d9772b25e))
* add refactor to changelog template ([`8738126`](https://github.com/mdtanker/polartoolkit/commit/8738126c46fa40bf44c8f3c98e379d314a6b133d))
* add mypy to dev deps ([`8102792`](https://github.com/mdtanker/polartoolkit/commit/810279299a717447652f9a7560ea04078f8c6207))
* add mypy make command ([`ed80102`](https://github.com/mdtanker/polartoolkit/commit/ed801021a2ec2ebead4cc7651325a7b99e2cdfe8))
* add run_notebook make command ([`1272361`](https://github.com/mdtanker/polartoolkit/commit/12723612dc6efd020a73cde15297689d7477e9dd))
* add types-request to mypy ([`6e14d17`](https://github.com/mdtanker/polartoolkit/commit/6e14d17847dd4ef7753503c88908ad1fb3758f95))
* ignore contrib guide in pre-commit ([`be24667`](https://github.com/mdtanker/polartoolkit/commit/be24667985e56628703e5d54f8b4af055c065f99))
* remove RTD mock imports ([`80d6d0f`](https://github.com/mdtanker/polartoolkit/commit/80d6d0f46fcd9991df59d7c1282847ef062e82fd))
* remove dynamic version ([`53f018b`](https://github.com/mdtanker/polartoolkit/commit/53f018bb6d37e91fa4dbfc892b77b8bdca9c52b3))
* update license notice file ([`c51b7b1`](https://github.com/mdtanker/polartoolkit/commit/c51b7b15826abb4e2525cc9a04551fff17893e93))
* add changelog template ([`855691e`](https://github.com/mdtanker/polartoolkit/commit/855691efa9403fc3ec6570c3556182401fca10f6))
* update project urls ([`4cf10f8`](https://github.com/mdtanker/polartoolkit/commit/4cf10f8facb43f3d356118b065918bfeb95dc45d))
* switches version management system

changes from setuptools_scm to python-semantic-release ([`46df13d`](https://github.com/mdtanker/polartoolkit/commit/46df13d314d9ec364dc541dfc45a6d826a37b31c))
* add pylint config ([`91dfa92`](https://github.com/mdtanker/polartoolkit/commit/91dfa926754ecfac1f67b29d6b83a74a538112c5))
* add ruff config ([`5688ab9`](https://github.com/mdtanker/polartoolkit/commit/5688ab992bea44a422dcd5cec9d6abca3565c12b))
* add mypy config ([`cd1805b`](https://github.com/mdtanker/polartoolkit/commit/cd1805b135e9701ca63ee136d5cc0b347365b802))
* move regions to src ([`c22281f`](https://github.com/mdtanker/polartoolkit/commit/c22281fbd547737f697066624052dc65fca64ae7))
* move utils to src ([`39e477d`](https://github.com/mdtanker/polartoolkit/commit/39e477d6115ccf1445591399b1f06b6eb4f934c5))
* move profile to src ([`0c6a014`](https://github.com/mdtanker/polartoolkit/commit/0c6a014cdd13d803d457f40c3180efba4dd3dd27))
* move maps to src ([`e2e561d`](https://github.com/mdtanker/polartoolkit/commit/e2e561dfd26ed246409ee1b00a9e02af4378a1a4))
* move fetch to src ([`5a915b6`](https://github.com/mdtanker/polartoolkit/commit/5a915b6439ea48e5155ff08fa92ff06d013b69d0))
* move init to src ([`d3f229f`](https://github.com/mdtanker/polartoolkit/commit/d3f229f7f144f95c3fe3b4ff9f03e3aba9770aad))
* remove test init ([`a381570`](https://github.com/mdtanker/polartoolkit/commit/a381570352279bc2cbe98324dadea6888d814eb8))
* add github/matchers file ([`99ed168`](https://github.com/mdtanker/polartoolkit/commit/99ed16885b8c2cdd14b2c8fd52c1e18246d65965))
* pre-commit ignore .github files ([`ffb36c9`](https://github.com/mdtanker/polartoolkit/commit/ffb36c99bd34f90347ce6ccb30ce26973bcdc342))
* add noxfile ([`6cac09d`](https://github.com/mdtanker/polartoolkit/commit/6cac09d7d46d9bf9a7359cab79e7316f5de7082e))
* update makefile ([`d619207`](https://github.com/mdtanker/polartoolkit/commit/d6192070f8dcb163ec577807cdeab100fa8b0a4a))
* add pre-commit config ([`8c33642`](https://github.com/mdtanker/polartoolkit/commit/8c33642542d752ebc28b135a0986fa9788573a1b))
* update gitignore ([`5912c52`](https://github.com/mdtanker/polartoolkit/commit/5912c52ac5ef20384df18d0106af0b0cdd0247af))
* move tests outside src ([`e7b30e9`](https://github.com/mdtanker/polartoolkit/commit/e7b30e9f23da359c0911c0c091fe2e9da1b3b87f))
### ✏️ Formatting
* formatting ([`429b998`](https://github.com/mdtanker/polartoolkit/commit/429b9984ec407e21ed07956b3491f97722a48b15))
* formatting ([`ddb0e42`](https://github.com/mdtanker/polartoolkit/commit/ddb0e42987e6d4557ecbe9ae9c1414c78525af00))
* fix spelling error ([`ea3ed50`](https://github.com/mdtanker/polartoolkit/commit/ea3ed501603d79afc34c14e561df08ad64bdbb95))
* formatting test_utils ([`cf83691`](https://github.com/mdtanker/polartoolkit/commit/cf83691db1711c66e7b5d68d3731ad6d89790b8c))
* formatting ([`e1e5aa8`](https://github.com/mdtanker/polartoolkit/commit/e1e5aa8f4cce375578c74590de23ce8af8e5db75))
* formatting ([`abbae80`](https://github.com/mdtanker/polartoolkit/commit/abbae80f153674078a188af85fda92dad36f22db))
* formatting ([`68f78aa`](https://github.com/mdtanker/polartoolkit/commit/68f78aa1238446ab114d757dd51207b1ab30d546))
* formatting ([`f9f82f8`](https://github.com/mdtanker/polartoolkit/commit/f9f82f8201e08212e61c431a9259075884f1b4c5))
### 📖 Documentation
* update notebooks ([`7851e53`](https://github.com/mdtanker/polartoolkit/commit/7851e53b9f4afa1cb592a1cbecf28ffb52aecfa7))
* rerun gallery examples ([`3c6f4c0`](https://github.com/mdtanker/polartoolkit/commit/3c6f4c0577f7db42538c46a8c3733964386615cb))
* rerun tutorials ([`e61324a`](https://github.com/mdtanker/polartoolkit/commit/e61324aab0a0e7bc53e2747a0aa831f119f32b2f))
* update cover_fig ([`1377bf0`](https://github.com/mdtanker/polartoolkit/commit/1377bf000ca4d37c0c9708806d7cd3c067e89226))
* fixes small issues ([`e8339ed`](https://github.com/mdtanker/polartoolkit/commit/e8339ed83aea8dcc0a0abc794328fd050fc23444))
* add/fix all docstrings ([`7c670d9`](https://github.com/mdtanker/polartoolkit/commit/7c670d97da47937ed9e127580f9ec39457fd894c))
* setup nbgallery for docs ([`48f061a`](https://github.com/mdtanker/polartoolkit/commit/48f061a78d5bf2eab1ffb2c3b05ba0525fbc0e12))
* update contrib guide ([`69663f9`](https://github.com/mdtanker/polartoolkit/commit/69663f907ab34467d52685863c4d2ca6be2641e4))
* update README and index ([`fea2b09`](https://github.com/mdtanker/polartoolkit/commit/fea2b09e764c57135791322dc8ce86bc83a0fd28))
* remove notebook output files ([`4a8c78b`](https://github.com/mdtanker/polartoolkit/commit/4a8c78b3c5866a740235fcea2fc373ce4d0416fe))
* add module descriptions to overview ([`92edec5`](https://github.com/mdtanker/polartoolkit/commit/92edec57f5356431b10430d83a075306b3ca81c2))
* pin RTD python version ([`f25810d`](https://github.com/mdtanker/polartoolkit/commit/f25810ddb2b2aca142b4f6499c9fb87935897bfa))
* move changelog ([`50c5439`](https://github.com/mdtanker/polartoolkit/commit/50c54392a5ab596e7e77e3fe607018ee92c6f889))
* add citing, overview, references ([`f27a893`](https://github.com/mdtanker/polartoolkit/commit/f27a893f62ea0a832ab5c3cfe6bb283f2f4eb85b))
* rename tutorials.md ([`5134420`](https://github.com/mdtanker/polartoolkit/commit/5134420d48c072a745468d8225e3352ca83127c2))
* rename gallery.md ([`4371781`](https://github.com/mdtanker/polartoolkit/commit/43717813fd60066cced567744d01ceef69476a73))
* update api docs with template ([`116f06e`](https://github.com/mdtanker/polartoolkit/commit/116f06e27de8431ebcfd2b82e54114b8042b776a))
* rename install file ([`4b97d66`](https://github.com/mdtanker/polartoolkit/commit/4b97d66dc5d08de824b62add59ca577161a6b94a))
* switch docs theme ([`c1a5d5a`](https://github.com/mdtanker/polartoolkit/commit/c1a5d5a00ce3f2579e291498818cf75e9d89bc3b))
* move contrib guide ([`d4e47ca`](https://github.com/mdtanker/polartoolkit/commit/d4e47ca386b873e9570da71f6ea203fe2c143b49))
* combine index and readme ([`1f4bfc1`](https://github.com/mdtanker/polartoolkit/commit/1f4bfc1d0c37980079a9d51d9d4572441acb820f))
* change RTD config ([`e17be26`](https://github.com/mdtanker/polartoolkit/commit/e17be2690af1e2b583f94211f244607b9e0579cd))
### 🚀 Features
* add grd2cpt and shading to profile map ([`2440c27`](https://github.com/mdtanker/polartoolkit/commit/2440c277aab205ac6463269e25976549f8005b37))
* add spacing option to default layers ([`69d72f4`](https://github.com/mdtanker/polartoolkit/commit/69d72f464b4b142eb860d2de49016871762fd79e))
* add lake vostok region ([`69b5ff6`](https://github.com/mdtanker/polartoolkit/commit/69b5ff66b4bcb4d5469f05e20014f01b51aea73d))
* pass scalebar kwargs to plot_grd ([`e733241`](https://github.com/mdtanker/polartoolkit/commit/e733241d15d229c0d3e7d006167db28fddb11e3e))
* add get_fetches function ([`b0312c2`](https://github.com/mdtanker/polartoolkit/commit/b0312c2254e5781668cf6fbba563e1dec00b473d))
###  🎨 Refactor
* fix issues with fetch tests

swaps pytest.approx with DeepDiff, adds ignore RuntimeWarning to many tests. ([`3b1bf49`](https://github.com/mdtanker/polartoolkit/commit/3b1bf497becf92d408ae701258051bb0491ad44c))
* switch regions from lists to tuples ([`65d7d92`](https://github.com/mdtanker/polartoolkit/commit/65d7d9234a4177f131d3e0e7c7e0415d25e71208))
* update optional deps import check ([`9926a59`](https://github.com/mdtanker/polartoolkit/commit/9926a59ff5bb1856057241c93588ed96cdc649dc))
* standardize preprocessing calls

increments filename variabls since they change type, and standardizes the format of the preprocessors. ([`4f5656f`](https://github.com/mdtanker/polartoolkit/commit/4f5656fa58ab155dffb00cdc114ce18fd04686d1))
### Other
*  ([`90302ba`](https://github.com/mdtanker/polartoolkit/commit/90302ba4e059c6688dad9468c42cb38c62ad6540))
*  ([`12a5299`](https://github.com/mdtanker/polartoolkit/commit/12a5299d590c3ce16e797fd3efd18ea8e7b2234c))
*  ([`c962e56`](https://github.com/mdtanker/polartoolkit/commit/c962e56ca0eca300b9f45bd4104861cd9fdf67dd))
*  ([`95ad63c`](https://github.com/mdtanker/polartoolkit/commit/95ad63c9af3c9b305fdbc97c984ac6bee952c3bf))
*  ([`30ea5aa`](https://github.com/mdtanker/polartoolkit/commit/30ea5aaea6cf599299e0cc7253eaf136b1e54a59))
*  ([`d8c821e`](https://github.com/mdtanker/polartoolkit/commit/d8c821ee79f146bc74171ae358a331815ba1ac7d))
*  ([`f1249f0`](https://github.com/mdtanker/polartoolkit/commit/f1249f0d7cb848f489bd25f8364d1693e704f886))


> **Note:**
>🚨
Everything above this point was generated automatically by Python Semantic Release.
Everything below is from prior to the implementation of Python Semaintic Release. 🚨


## Between v0.0.6 and v0.1.0

### 💫 Highlights
* dropped support for Python 3.8 in PR #140
* several new datasets!
* added `robust` option to get_min_max(), and mapping functions
* lots of new customization options (kwargs) for plotting functions
* several bug fixes

### 🚀 Features

#### New datasets in `Fetch`
* Ice Mass Changes from Smith et al. 2020
* Basal Melt Rates from Adusumulli et al. 2020
* Faults and geologic unit shapefiles from GEOMAP (Cox et al. 2023)
* ADMAP-2 magnetics compilation
* ROSETTA-Ice airborne magnetics for the Ross Ice Shelf from Tinto et al. 2019
* BedMachine updated to v3

#### New functions in `regions`
* `get_regions()``
    * list all available regions

### 📖 Documentation

### ⛔ Maintenance
* new URL for fetching ADMAP1 magnetic data
* added Harmonica as a dependency for ADMAP-2 geosoft grid conversion
* fix old binder links
* added `north_arrow()` function call to `plot_grd()` with keyword `add_north_arrow`
* fixed issues with clipping of grid for plotting colorbar histogram

### 🧑‍🤝‍🧑 Contributors
[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.6

### 💫 Highlights
* Switched from Poetry to Setuptools
* Can install with conda
* Eased the dependency constraints

### ⛔ Maintenance
* renamed various util functions

### 🧑‍🤝‍🧑 Contributors
[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.4

### 💫 Highlights
* New mapping function `antarctic_plots.maps`
* Pre-set regions for commonly plotted areas
* Added Gallery examples
* Created a Binder environment
* More datasets included in `fetch`

### 🚀 Features

#### New module `Maps`

* plot_grd

#### New datasets in `Fetch`

* bedmachine
* geothermal

#### New functions in `Utils`

* alter_region
* coherency
* grd_compare
* grd_trend
* make_grid
* raps
* set_proj

### 📖 Documentation

* Added `Tutorials` and `Gallery examples` to the docs
* Added tutorial for modules `fetch` and `region`

### ⛔ Maintenance
* Closed [Issue #6](https://github.com/mdtanker/antarctic_plots/issues/6): Create gallery examples
* Closed [Issue #9](https://github.com/mdtanker/antarctic_plots/issues/9): Code formatting
* Closed [Issue #13](https://github.com/mdtanker/antarctic_plots/issues/13): Specify dependency version
* Closed [Issue #15](https://github.com/mdtanker/antarctic_plots/issues/15): Add inset map of Antarctica
* Closed [Issue #16](https://github.com/mdtanker/antarctic_plots/issues/16): Add Welcome Bot message to first time contributors
* Closed [Issue #20](https://github.com/mdtanker/antarctic_plots/issues/20): Add options to use the package online
* Closed [Issue #25](https://github.com/mdtanker/antarctic_plots/issues/25): Add GHF data to fetch module
* Closed [Issue #26](https://github.com/mdtanker/antarctic_plots/issues/26): Add BedMachine Data to fetch
* Closed [Issue #27](https://github.com/mdtanker/antarctic_plots/issues/27): fetch.bedmap2 issue with xarray
* Closed [Issue #28](https://github.com/mdtanker/antarctic_plots/issues/28): Set region strings for commonly plotted areas
* Closed [Issue #22](https://github.com/mdtanker/antarctic_plots/issues/22): Create Zenodo DOI

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.3

### 💫 Highlights
* Finally succeeded in building the docs!

### 📖 Documentation

* Added `make build-docs` to execute and overwrite .ipynb to use in docs, since `PyGMT` can't be included in dependencies and therefore RTD's can't execute the .ipynb's.

### ⛔ Maintenance

* Closed [Issue #7](https://github.com/mdtanker/antarctic_plots/issues/7)

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.2

### 💫 Highlights
* Created a [website for the documentation!](https://antarctic_plots.readthedocs.io/en/latest/installation.html#)

* Added `NumPy` formatted docstrings to the modules

* Wrote contribution guide, which outlines the unique case of publishing a package with dependencies which need C packages, like `PyGMT` (`GMT`) and `GeoPandas` (`GDAL`).

* Added `Tips` for generating shapefiles and picking start/end points

### 📖 Documentation

* Re-wrote docstrings to follow `NumPy` format.
* Added type-hints to docstrings.

### ⛔ Maintenance

* Closed [Issue #13](https://github.com/mdtanker/antarctic_plots/issues/13)
* Closed [Issue #9](https://github.com/mdtanker/antarctic_plots/issues/9)
* Closed [Issue #2](https://github.com/mdtanker/antarctic_plots/issues/2)


### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.1

### 💫 Highlights
* also probably should have been published to TestPyPI 🤦♂️

### 🚀 Features

* Added a Makefile for streamlining development, publishing, and doc building.
* Added license notifications to all files.


### 📖 Documentation

* Used `Jupyter-Book` structure, with a table of contents (_toc.yml) and various markdown files.
* added `Sphinx.autodoc` to automatically include API documentation.


### ⛔ Maintenance

* Looks of issues with the Poetry -> Jupyter-Books -> Read the Docs workflow
* Poetry / RTD don't like `PyGMT` or `GeoPandas` since they both rely on C packages which can't be installed via pip (`GMT` and `GDAL`). Next release should fix this.


### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.0

* 🎉 **First release of Antarctic-plots** 🎉

* should have been published to TestPyPI 🤦♂️

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)
