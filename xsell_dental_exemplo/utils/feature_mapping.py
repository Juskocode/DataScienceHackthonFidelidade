import pandas as pd

# Function to create Feature Mapping for OneHotEncoder
def ohe_feature_mapping(one_hot_encoder):
    feature_mapping = {}
    for i, col in enumerate(one_hot_encoder.feature_names_in_):
        for category in one_hot_encoder.categories_[i]:
            if str(category) not in ('nan','0'):
                output_col_name = f"{col}_{category}"
                feature_mapping[output_col_name] = [(col,'OneHotEncoder')]
    return feature_mapping

# Function to keep only the final variables with all the history of transformations
def clean_feature_mapping(feature_mapping):
    # keep all transformations
    final_features = list(feature_mapping.keys())
    keys_to_delete = []
    # it just need 1 cycle because the features are ordered by creation time
    for i, f in enumerate(final_features):
        for j, f2 in enumerate(feature_mapping[f]):
            if f2[0] in final_features[:i]:  # check only features created after
                #             feature_mapping[f][j] = feature_mapping[f2[0]
                feature_mapping[f] = feature_mapping[f][:j] + feature_mapping[f2[0]] + feature_mapping[f][j:]
                keys_to_delete.append(f2[0])

# TIVEMOS QUE APAGAR PORQUE HÁ VAIÁVEIS INTERMEDIAS E QUE TAMBÉM SÃO FINAIS
#    for f in set(keys_to_delete):
#        del feature_mapping[f]

    for k, v in feature_mapping.items():
        feature_mapping[k] = pd.Series(feature_mapping[k]).drop_duplicates().tolist()
    return feature_mapping


# Function to group consecutive transformations with the same step name
def group_transformations(transform_list):
    grouped = []
    current_group = []
    current_step = None

    for variable, step in transform_list:
        if step == current_step:
            current_group.append(variable)
        else:
            if current_group:
                grouped.append((current_group, current_step))
            current_group = [variable]
            current_step = step

    if current_group:
        grouped.append((current_group, current_step))

    return grouped


# Function to print the diagram
def print_diagram(feature_mapping, feature=""):
    """
    Prints a diagram showing the sequence of transformations applied to variables
    to produce a final variable.

    Parameters
    ----------
    feature_mapping : dict
        Dictionary where the key is the final variable, and the value is a list of
        (variable_name, step_name) tuples or lists showing each transformation step.

    feature : str or list of str, optional
        The specific variable(s) to generate the diagram for. If not provided,
        diagrams for all variables in `feature_mapping` will be printed.

    Returns
    -------
    None
        This function prints the transformation diagram to the console.
    """
    if feature == "":
        mapping_dict = feature_mapping
    elif type(feature) is str:
        mapping_dict = {feature: feature_mapping[feature]}
    elif type(feature) is list:
        mapping_dict = {key: feature_mapping[key] for key in feature if key in feature_mapping.keys()}

    for final_variable, transform_list in mapping_dict.items():
        print(f"Transformation diagram for: \033[1m{final_variable}\033[0m")
        grouped_transformations = group_transformations(transform_list)

        previous_layer = None
        for i, (variables, step) in enumerate(grouped_transformations):

            layer = " | ".join([f"\033[1m{var}\033[0m" for var in variables])

            # Print arrows to the previous layer
            if previous_layer:
                print(" " * int(20) + "|")
                print(" " * int(20) + "V")
                print(f"Output / Input variable(s): [{layer}]")
            else:
                print(f"Original variable(s): [{layer}]")
            print(" " * int(20) + "|")
            print(" " * int(20) + "V")
            print(f"Layer {i + 1}: [\033[1m{step}\033[0m]")

            previous_layer = variables

        print(" " * int(20) + "|")
        print(" " * int(20) + "V")
        print(f"Final variable: \033[1m{final_variable}\033[0m")
        print("-" * 50)