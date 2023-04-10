def create_bagging_pickle(model, model_name: str, window_size: int, h_step: int, testing: list, total: list, lags: int,
                      scaler, level_grid: str):
    data = {'model': model['model'], 'model_name': model_name, 'window_size': window_size, 'h_step': h_step,
            'training_sample': model['training_sample'], 'validation_sample': model['validation_sample'],
            'testing_sample': testing, 'total_sample': total, 'lags': lags, 'scaler': scaler,
            'level_grid': level_grid, 'indices': model['indices'], 'number_of_models': len(model['model'])}

    return data


def create_monolithic_pickle(model, model_name, window_size: int, h_step: int, training: list, validation: list, testing: list,
                             total: list, lags: int, scaler, level_grid: str):
    from c1copy import calculate_model_accuracy

    data = dict(model=model, model_name=model_name, window_size=window_size, h_step=h_step, training_sample=training,
                validation_sample=validation, testing_sample=testing, total_sample=total, lags=lags, scaler=scaler,
                level_grid=level_grid, number_of_models=1,
                mse=calculate_model_accuracy(testing[:, -1], model.predict(testing[:, lags[: -1]]), measure='mse'))

    return data


def save_model(model, model_name: str, ws: int, h_step: int, training: list, testing: list,
               total: list, lags: int, scaler, level_grid: str, file_path: str, validation=None):
    if validation is None:
        validation = []

    if level_grid == 'default' or level_grid == 'hard':
        save_pickle(create_monolithic_pickle(model, model_name, ws, h_step, training, validation, testing, total, lags,
                                             scaler, level_grid), file_path)

    if level_grid == 'bagging':
        df = create_bagging_pickle(model, model_name, ws, h_step, testing, total, lags, scaler, level_grid)

        for i in range(0, df['number_of_models']):
            df['model'] = model['model'][i]
            df['training_sample'] = model['training_sample'][i]
            df['validation_sample'] = model['validation_sample'][i]
            df['indices'] = model['indices'][i]

            save_pickle(df, file_path[:-2] + level_grid + str(i) + str(ws))


def save_pickle_result(predl, targetl, names, list_y_models_sequence, filename_pickle):
    from c1copy import calculate_model_accuracy

    df_pickle = {'y_true_testing': targetl, 'y_pred_testing': predl,
                 'models_name': names,
                 'models': list_y_models_sequence,
                 'mse': calculate_model_accuracy(targetl, predl, 'mse')}

    save_pickle(df_pickle, filename_pickle)


def save_pickle(df_pickle, filename_pickle: str):
    from pickle import dump

    dump(df_pickle, open('pickle/' + filename_pickle + '.pkl', 'wb'))


def load_pickle(filename_pickle: str):
    from pickle import load

    return load(open('pickle/' + filename_pickle + '.pkl', 'rb'))
