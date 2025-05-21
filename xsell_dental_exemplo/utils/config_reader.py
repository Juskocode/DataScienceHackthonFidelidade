import logging
from typing import Dict, List, Optional  # , Any, Dict, Union

import yaml
from pydantic import BaseModel, validator

from datetime import datetime

# Configure logging for the external script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
formatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class CreateUniverseConfig(BaseModel):
    lob: str
    client_type: str
    start_month: int
    end_month: int
    id_column: str
    target1_universe: str
    car_filters: Optional[List[str]] = []
    par_filters: Optional[List[str]] = []
    IND_PSIN_PCOL_ENI : List[str]
    list_of_products: Optional[List[str]] = []
    grounds_for_cancellation : Optional[List[str]] = []
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_yearmonth()
        self.uppercase()
        
    def convert_yearmonth(self):
        self.start_month = datetime.strptime(str(self.start_month), "%Y%m")
        self.end_month = datetime.strptime(str(self.end_month), "%Y%m")
        
    def uppercase(self):
        self.lob = self.lob.upper()
        self.client_type=self.client_type.upper()
        
    def get_params(self):
        """Retrieve all input parameters."""
        return self.__dict__        

class VariablesUniverseConfig(BaseModel):
    variables: Optional[Dict[str, List[str]]] = None

class PrepareDataConfig(BaseModel):
    input_file: Optional[str] = ""
    id_column: str
    version: str
    test_size: Optional[float] = 0.1
    columns_to_drop: Optional[List[str]] = None

    @validator("version")
    def validate_version(cls, v):
        if not v.startswith("v"):
            raise ValueError("Version number must start with 'v'.")
        version_number = v[1:]
        try:
            float(version_number)
        except ValueError:
            raise ValueError("Invalid version number format.")
        return v

    def get_params(self):
        """Retrieve all input parameters."""
        return self.__dict__


class TrainingConfig(BaseModel):
    train_type: str
    id_column: str
    version: str
    old_version: Optional[str] = None
    months: List[int]
    column_aggregation: List[bool]
    flag_creation: List[bool]
    binning: List[bool]
    outlier_cleaning: List[bool]
    standard_scaler: Optional[bool] = True
    overwrite_saved_features: Optional[bool] = True
    calibrated_classifier: Optional[bool] = True
    fixed_features: Optional[List[str]] = []
    n_features: Optional[List[str]] = []
    algorithms: Optional[List[str]] = []
    hyperparameters: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.algorithms is not None and self.algorithms != []:
            self.process_algorithms()

    def process_algorithms(self):
        ignore_models = []
        for model_name, param_grid in self.hyperparameters.items():
            if model_name in self.algorithms:
                for hyperparameter, values in param_grid.items():
                    self.hyperparameters[model_name][hyperparameter] = [
                        self._convert_to_tuple_or_none(v) for v in values
                    ]
            else:
                ignore_models.append(model_name)
        for model_name in ignore_models:
            del self.hyperparameters[model_name]

    @validator("train_type")
    def validate_train_type(cls, v):
        valid_values = ["train", "retrain", "recalibration"]
        if v not in valid_values:
            raise ValueError(f"'train_type' must be one of the following: {', '.join(valid_values)}")
        return v

    @validator("version", "old_version")
    def validate_version(cls, v):
        if not v.startswith("v"):
            raise ValueError("Version number must start with 'v'.")
        version_number = v[1:]
        try:
            float(version_number)
        except ValueError:
            raise ValueError("Invalid version number format.")
        return v

    @validator(
        "column_aggregation",
        "flag_creation",
        "binning",
        "outlier_cleaning",
        pre=True,
        each_item=True,
    )
    def validate_list_of_bools(cls, v):
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False
        return v

    @validator(
        "standard_scaler",
        "overwrite_saved_features",
        "calibrated_classifier",
        pre=True,
    )
    def validate_bool(cls, v):
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False
        return v

    @validator("n_features")
    def validate_n_features(cls, v):
        for item in v:
            if item != "automatic":
                try:
                    int(item)
                except ValueError:
                    raise ValueError("All items in 'n_features' must be integers unless 'automatic' is present.")
        return v

    def _convert_to_tuple_or_none(self, v):
        if v == "None":
            return None
        elif isinstance(v, str) and v.startswith("(") and v.endswith(")"):
            return eval(v)
        else:
            return v

    def _false_first(self, test_list):
        if len(test_list) == 2:
            if test_list[0] is True:
                test_list.reverse()
        return test_list

    def get_training_transformer_grid(self):
        """Generate combinations of training transformations."""
        training_grid = {}
        training_grid["column_aggregation"] = self._false_first(self.column_aggregation)
        training_grid["flag_creation"] = self._false_first(self.flag_creation)
        training_grid["binning"] = self._false_first(self.binning)
        training_grid["outlier_cleaning"] = self._false_first(self.outlier_cleaning)
        return training_grid

    def get_params(self):
        """Retrieve all input parameters."""
        return self.__dict__


class ScoresMonitoringConfig(BaseModel):
    mes_scores: int 
    id_column: str
    version: str
    MODEL_TYPE: str 
    MODEL_LOB: str

    REVIEW_PERIODICITY: Optional[str] = None

    THRESHOLD_GREEN_6m: Optional[float] = None #threshold para leads verdes - 6 meses
    THRESHOLD_YELLOW_6m: Optional[float] = None
    THRESHOLD_GREEN_12m: Optional[float] = None
    THRESHOLD_YELLOW_12m: Optional[float] = None
    PERCENTIL_YELLOW: Optional[int] = None
    PERCENTIL_RED: Optional[int] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_yearmonth()
        self.uppercase()
        
    def convert_yearmonth(self):
        self.mes_scores = datetime.strptime(str(self.mes_scores), "%Y%m")
        
    def uppercase(self):
        self.MODEL_TYPE = self.MODEL_TYPE.upper()
        self.MODEL_LOB=self.MODEL_LOB.upper()

    @validator("version")
    def validate_version(cls, v):
        if not v.startswith("v"):
            raise ValueError("Version number must start with 'v'.")
        version_number = v[1:]
        try:
            float(version_number)
        except ValueError:
            raise ValueError("Invalid version number format.")
        return v

    @validator("REVIEW_PERIODICITY")
    def validate_review_periodicity(cls, value):
        valid_periodicities = ["semestral", "anual"]
        if value.lower() not in valid_periodicities:
            raise ValueError(f"REVIEW_PERIODICITY must be one of the following: {', '.join(valid_periodicities)}")
        return value

    def get_params(self):
        """Retrieve all input parameters."""
        return self.__dict__

class DataAggregationConfig(BaseModel):
    new_file: str
    historical_file: Optional[str] = None
    target_var_name: Optional[str] = "TARGET"
    id_column: str
    universe_months: Optional[int] = 12

    def get_params(self):
        """Retrieve all input parameters."""
        return self.__dict__


class RegisterModelConfig(BaseModel):
    version: str
    run_id: Optional[str] = None

    def get_params(self):
        """Retrieve all input parameters."""
        return self.__dict__

    @validator("version")
    def validate_version(cls, v):
        if not v.startswith("v"):
            raise ValueError("Version number must start with 'v'.")
        version_number = v[1:]
        try:
            float(version_number)
        except ValueError:
            raise ValueError("Invalid version number format.")
        return v


def read_config(file_path: str):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)

    if "variables_universe_config" not in file_path.name:
        logging.info("Input parameters: %s", config_data)

    if "create_universe_config" in file_path.name:
        return CreateUniverseConfig(**config_data)
    
    elif "variables_universe_config" in file_path.name:
        return VariablesUniverseConfig(variables = config_data)

    elif "prepare_data_config" in file_path.name or "predict_config" in file_path.name:
        return PrepareDataConfig(**config_data)

    elif "training_config" in file_path.name:
        return TrainingConfig(**config_data)

    elif 'scores_monitoring' in file_path.name:
        return ScoresMonitoringConfig(**config_data)

    elif "data_aggregation_config" in file_path.name:
        return DataAggregationConfig(**config_data)

    elif "register_model_config" in file_path.name:
        return RegisterModelConfig(**config_data)
