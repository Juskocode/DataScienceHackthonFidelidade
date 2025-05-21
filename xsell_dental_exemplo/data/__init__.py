
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # Importação do PdfPages

#%%
from xsell_dental_exemplo.utils import exploratory_data_analysis as eda
from xsell_dental_exemplo.utils.config_reader import read_config
from xsell_dental_exemplo.utils import data_processing as dp
from xsell_dental_exemplo.utils import universe_creation as uc
#%%
from sklearn.model_selection import train_test_split
#%%
PROJECT_PATH = Path(__file__).parent
# PROJECT_PATH = Path(globals()['_dh'][0]).parent
print(PROJECT_PATH)
DATA_PATH = PROJECT_PATH / "data"
print(DATA_PATH)
LABEL_PATH = DATA_PATH / "metadata"
print(LABEL_PATH)
INPUT_DATA_PATH = PROJECT_PATH / "data" / "Input Data"
print(INPUT_DATA_PATH)
AUX_REPORT_PATH = PROJECT_PATH / "report" / "Auxiliary Reports"
print(AUX_REPORT_PATH)
MODEL_PATH = PROJECT_PATH / "models"
print(MODEL_PATH)
CONFIG_PATH = PROJECT_PATH / "configs"
print(CONFIG_PATH)
REPORT_PATH = PROJECT_PATH / "report"
print(REPORT_PATH)