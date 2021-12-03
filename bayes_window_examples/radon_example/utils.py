import pandas as pd

def load_radon():
    # Import radon data
    srrs2 = pd.read_csv("srrs2.dat", error_bad_lines=False)
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state == "MN"].copy()

    # Next, obtain the county-level predictor, uranium, by combining two variables.

    srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    cty = pd.read_csv("cty.dat", error_bad_lines=False)
    cty_mn = cty[cty.st == "MN"].copy()
    cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

    # Use the merge method to combine home- and county-level information in a single DataFrame.
    srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
    srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
    # srrs_mn.county = srrs_mn.county.map(str.strip)
    # mn_counties = srrs_mn.county.unique()
    # counties = len(mn_counties)
    # county_lookup = dict(zip(mn_counties, range(counties)))

    # Finally, create local copies of variables.

    # county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
    # radon = srrs_mn.activity
    # srrs_mn["log_radon"] = np.log(radon + 0.1).values
    # floor = srrs_mn.floor.values

    return pd.DataFrame({'county': srrs_mn.county, 'radon': srrs_mn.activity, 'floor': srrs_mn.floor.values})
