import numpy as np
from folktables import ACSDataSource, ACSEmployment

def test_archive_download():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', use_archive=True)
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    assert features.shape == (378817, 16)

def test_archive_definitions_download():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', use_archive=True)
    definition_df = data_source.get_definitions(download=True)

    # test some definition_df properties
    assert len(definition_df.columns) == 7
    assert (np.isin(definition_df[0].unique(),['NAME','VAL'])).all()
    assert (np.isin(definition_df[2].unique(), ['C', 'N'])).all()

if __name__ == "__main__":

    test_archive_download()
    test_archive_definitions_download()
