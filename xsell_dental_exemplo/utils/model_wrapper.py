
from mlflow.pyfunc import PythonModel, PythonModelContext
import logging


class ModelWrapper(PythonModel):

    def load_context(self, context: PythonModelContext):
        import pickle

        ## LOAD YOUR PROCESSORS HERE
        # self._first_processor = pickle.load(open(context.artifacts["first_processor"], "rb"))
        # self._second_processor = pickle.load(open(context.artifacts["second_processor"], "rb"))
        
        ## LOAD YOUR MODEL HERE
        self._model = pickle.load(open(context.artifacts["model"], "rb"))


    def predict(self, context: PythonModelContext, input_tuple):

        data, id_column = input_tuple
        special_columns = [id_column, "ANO_MES"]


        ## APPLY ALL DATA TRANSFORMATIONS
        
        ## END OF DATA TRANSFORMATIONS

        
        logging.info("Loading model and generating predictions...\n")
        pred = self._model.predict_proba(X_test)
   
        X_test["Prediction_0"] = pred[:, 0]
        X_test["Prediction_1"] = pred[:, 1]    
        
    
        X_special = X_test.reset_index()[special_columns + ['Prediction_0', 'Prediction_1']]
                           
        predictions = X_special.merge(y_test.reset_index(), on=(special_columns), how="left")
        
        
        logging.info(f"Shape of input Test data: {data.shape}")
        logging.info(f"Shape of transformed Test data: {X_test.shape}")
        logging.info(f"Number of Targets (y_test): {y_test.reset_index().iloc[:, 2].shape}")
        logging.info(f"Shape of predictions (final) tabel: {predictions.shape}")

        
        # output_file_name = Path(output_path).stem
        # output_file_path = os.path.join(output_path, output_file_name + ".csv")
        # test.csv(output_file_path, index=False)

        return predictions