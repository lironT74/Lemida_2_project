from log_linear_memm import Log_Linear_MEMM
import pickle

def validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list,
             small_model):
    model_index = start_index
    for threshold in thresholds:
        for fix_threshold in fix_thresholds:
            for lam in lambdas:
                for fix_weights in fix_weights_list:
                    if small_model:
                        _, accuracy_comp, iterations = evaluate(train_path, model_index, threshold, fix_threshold, lam,
                                                                maxiter, fix_weights, small_model)
                        write_report_line(report_path, model_index, threshold, fix_threshold, lam, maxiter, iterations,
                                          fix_weights, acc_comp=accuracy_comp)
                        print('model' + str(model_index) + ' comp2 accuracy=' + str(accuracy_comp))
                    else:
                        accuracy_test, accuracy_comp, iterations = evaluate(train_path, model_index, threshold,
                                                                            fix_threshold, lam, maxiter, fix_weights,
                                                                            small_model)
                        write_report_line(report_path, model_index, threshold, fix_threshold, lam, maxiter, iterations,
                                          fix_weights, acc_comp=accuracy_comp, acc_test=accuracy_test)
                        print('model' + str(model_index) + ' comp1 accuracy=' + str(accuracy_comp))
                    model_index += 1


def evaluate(train_path, model_index, threshold, fix_threshold, lam, maxiter, fix_weights, small_model):
    # creating and training a model
    model = Log_Linear_MEMM(threshold, fix_threshold, lam, maxiter, fix_weights)
    model.fit(train_path, iprint=-1)
    iterations = model.iter

    # saving trained model
    model.save('model' + str(model_index))

    if small_model:
        # predicting and saving predictions
        predictions_comp2 = model.predict('data/comp2_nltk_tagged.wtag', beam_size=2)
        with open('dumps/model' + str(model_index) + '_comp2_predictions.pkl', 'wb') as f:
            pickle.dump(predictions_comp2, f)

        # calculating accuracy
        accuracy_comp2_tagged = Log_Linear_MEMM.accuracy('data/comp2_nltk_tagged.wtag', predictions_comp2)
        return None, accuracy_comp2_tagged, iterations

    else:
        # predicting and saving predictions
        predictions_test1 = model.predict('data/test1.wtag', beam_size=2)
        with open('dumps/model' + str(model_index) + '_test1_predictions.pkl', 'wb') as f:
            pickle.dump(predictions_test1, f)
        predictions_comp1 = model.predict('data/comp1_nltk_tagged.wtag', beam_size=2)
        with open('dumps/model' + str(model_index) + '_comp1_predictions.pkl', 'wb') as f:
            pickle.dump(predictions_comp1, f)

        # calculating accuracy
        accuracy_test1 = Log_Linear_MEMM.accuracy('data/test1.wtag', predictions_test1)
        accuracy_comp1_tagged = Log_Linear_MEMM.accuracy('data/comp1_nltk_tagged.wtag', predictions_comp1)
        return accuracy_test1, accuracy_comp1_tagged, iterations


def write_report_header(report_path, small_model):
    with open(report_path, 'w') as report:
        report.write('model_index,threshold,fix_threshold,lam,maxiter,iterations,fix_weights1,' +
                     'fix_weights2,fix_weights3,fix_weights4,acc_comp')
        if not small_model:
            report.write(',acc_test')
        report.write('\n')


def write_report_line(report_path, model_index, threshold, fix_threshold, lam, maxiter, iterations, fix_weights,
                      acc_comp, acc_test=False):
    with open(report_path, 'a') as report:
        report.write(str(model_index) + ',')
        report.write(str(threshold) + ',')
        report.write(str(fix_threshold) + ',')
        report.write(str(lam) + ',')
        report.write(str(maxiter) + ',')
        report.write(str(iterations) + ',')
        report.write(str(fix_weights[0]) + ',')
        report.write(str(fix_weights[1]) + ',')
        report.write(str(fix_weights[2]) + ',')
        report.write(str(fix_weights[3]) + ',')
        report.write(str(acc_comp))
        if acc_test:
            report.write(',' + str(acc_test))
        report.write('\n')
