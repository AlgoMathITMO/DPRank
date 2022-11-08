import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

class plots:
    
    def scatter_plot(sliced_x: pd.DataFrame, sliced_y: pd.DataFrame, days: np.array, train_info: str, fig_name: str, save: bool, folder: str):
        fig, axes = plt.subplots(3, 5)
        fig.set_size_inches(31, 22)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        for ax in axes.flat[len(days):]:
            ax.set_visible(False)
        for day, ax in zip(days, axes.flatten()):
            ax.scatter(sliced_x[day], sliced_y[day])
            ax.set_ylabel('Demand')
            ax.set_xlabel('Price')
            ax.set_title(train_info[:4]+', '+ str(day) + ' days before departure'+' ('+train_info[5:]+'s) ')
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid()
        if save==True:
            plt.savefig(folder+train_info+' '+fig_name+'.pdf', transparent=True, bbox_inches='tight')
            
    def predicted_scatter_plot(sliced_x: pd.DataFrame, sliced_y: pd.DataFrame, sliced_predicted_y, days: np.array, train_info: str, fig_name: str, save: bool, folder: str):
        fig, axes = plt.subplots(3, 5)
        fig.set_size_inches(31, 22)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        for ax in axes.flat[len(days):]:
            ax.set_visible(False)
        for day, ax in zip(days, axes.flatten()):
            ax.scatter(sliced_x[day], sliced_y[day])
            ax.plot(sliced_x[day], sliced_predicted_y[day], color='green')
            ax.set_ylabel('Demand')
            ax.legend(['Predicted demand and price', 'Linearized demand and price'])
            ax.set_xlabel('Price')
            ax.set_title(train_info[:4]+', '+ str(day) + ' days before departure'+' ('+train_info[5:]+'s) ')
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid()
        if save==True:
            plt.savefig(folder+train_info+' '+fig_name+'.pdf', transparent=True, bbox_inches='tight')
            
    def predicted_relined_scatter_plot(sliced_x: pd.DataFrame, sliced_y: pd.DataFrame, sliced_predicted_y, days: np.array, train_info: str, fig_name: str, train_type: str, save: bool, folder: str):
        fig, axes = plt.subplots(3, 5)
        fig.set_size_inches(31, 22)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        for ax in axes.flat[len(days):]:
            ax.set_visible(False)
        for day, ax in zip(days, axes.flatten()):
            sort_prices_mask=np.argsort(np.array(sliced_x[day]))
            sorted_y=np.array(sliced_predicted_y[day])[sort_prices_mask]
            ax.scatter(sliced_x[day], sliced_y[day])
            ax.plot(sliced_x[day][sort_prices_mask], pd.Series(sorted_y).interpolate(method='linear', axis=0, limit_direction='both', limit=100).values, color='green')
            ax.set_ylabel('Demand')
            ax.legend(['Demand approximation', 'Demand'])
            ax.set_xlabel('Price')
            ax.set_title(train_type+', '+ str(day) + ' days before departure'+' ('+train_info[5:]+'s) ')
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid()
        if save==True:
            plt.savefig(folder+train_info+' '+fig_name+'.pdf', transparent=True, bbox_inches='tight')
            
    def eps_hists(sliced_x: pd.DataFrame, days: np.array, train_info: str, fig_name: str, train_type: str, save: bool, folder: str):
        fig, axes = plt.subplots(3, 5)
        fig.set_size_inches(31, 22)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        for ax in axes.flat[len(days):]:
            ax.set_visible(False)
        for day, ax in zip(days, axes.flatten()):
            adf_results_dict=dict()
            interp_adf_results_dict=dict()
            sns.histplot(sliced_x[day][~np.isnan(sliced_x[day])], ax=ax, element="step", kde=False, alpha=1)
            ax.set_ylabel('Counts')
            ax.set_xlabel('\u03B4')
            ax.set_title(train_type+', '+ str(day) + ' days before departure'+' ('+train_info[5:]+'s) ')
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid()
            adf_result = adfuller(sliced_x[day][~np.isnan(sliced_x[day])])
            adf_results_dict[day]=adf_result
            if (adf_result[0]<=adf_result[4]['5%']) and (adf_result[1]<=0.05):
                interp_adf_results_dict[day]='H0 can be rejected'
            else:
                interp_adf_results_dict[day]='H0 cannot be rejected'
            tl = ((ax.get_xlim()[1] - ax.get_xlim()[0])*0.010 + ax.get_xlim()[0], \
                  (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95 + ax.get_ylim()[0])
            ax.text(tl[0], tl[1], r"ADF result: {}".format(interp_adf_results_dict[day]))
        if save==True:
            plt.savefig(folder+train_info+' '+fig_name+'.pdf', transparent=True, bbox_inches='tight')

class processing:
    
    def mean_duplicate_prices(sliced_data: pd.Series, sliced_tardet: pd.Series)->pd.DataFrame:
    
        '''The function accepts slices by price and demand, for a certain train, 
        day of the week and day before departure, the output is a pivot table in
        which demand values are averaged for which prices are the same. The class 
        and date columns store all classes and dates for which the average was performed.'''
    
        df=pd.DataFrame(data=np.array([sliced_data.values, sliced_tardet.values]).T, 
                    columns=['prices', 'demand'], index=sliced_data.index, dtype=object).reset_index()
        if len(df['prices'].dropna())==0:
            piv=df
        else:
            piv=df.pivot_table(index=['prices'], sort = False, dropna=False, 
                                         values=['demand', 'date', 'class'], 
                                         aggfunc={'demand': np.nanmean, 
                                                  'date': lambda x: x, 
                                                  'class': ','.join}).reset_index()
        return piv

    def flat_grouped_classes(pivot_data: pd.DataFrame)->pd.DataFrame:
    
        '''The function takes a pivot table with prices and demand 
        from the mean_duplicate_prices function and straightens the
        class and date groups. The cell with the averaged demand retains 
        the class most frequently encountered during the average and the 
        corresponding date. The rest of the classes and dates are added to
        the end of the frame, with empty price and demand values.'''
    
        for row in pivot_data.T:
            class_row=list(pivot_data['class'][row].split(','))
            date_row=[pivot_data['date'][row]]
            str_date_row=list(pivot_data['date'][row].strftime('%Y-%m-%d'))
            if len(class_row)>1:
                count_classes=Counter(class_row)
                max_val = max(count_classes.values())
                max_class = [k for k, v in count_classes.items() if v == max_val][0]
                class_num=[class_num for class_num, 
                           class_name in enumerate(class_row) 
                           if class_name==max_class][0]
                pivot_data.loc[pivot_data.index[row], 'class']=max_class
                pivot_data.loc[pivot_data.index[row], 'date']=date_row[0][class_num]
                del class_row[class_num], str_date_row[class_num]
                for cl, dt in zip(class_row, str_date_row):
                    pivot_data=pd.concat([pivot_data, pd.DataFrame([[np.nan, cl, dt,  np.nan]], columns=pivot_data.columns)])
                    pivot_data['date']=pivot_data['date'].apply(pd.to_datetime)
                    pivot_data.reset_index(drop = True , inplace = True)
        return pivot_data
    
    def linearize_vars(slised_data: pd.DataFrame, sliced_labels: pd.DataFrame)->(pd.DataFrame, pd.Series, pd.DataFrame):
        min_data=slised_data.min(axis=0)
        lin_data=slised_data-min_data+1.0
        log_lin_data=lin_data.apply(np.log, axis=0)
        shift_labels=sliced_labels+1.0
        log_shift_labels=shift_labels.apply(np.log, axis=0)
        log_lin_data=log_lin_data.mask(np.isinf(log_lin_data)).fillna(np.nan)
        log_shift_labels=log_shift_labels.mask(np.isinf(log_shift_labels)).fillna(np.nan)
        log_lin_data[np.isnan(log_shift_labels)]=np.nan
        log_shift_labels[np.isnan(log_lin_data)]=np.nan
        return lin_data, log_lin_data, min_data, shift_labels, log_shift_labels
    
class model_evaluation:
    
    def r2_simple(labels: list, labels_pred: list)->np.float64:
        if not(np.nanvar(labels)==0):
            r2=1.-(np.nanvar(labels-labels_pred)/np.nanvar(labels))
        else:
                r2=np.nan
        return r2

    def validation_model(data: tuple, labels: tuple):    
        data=data.dropna().values
        labels=labels.dropna().values
        line_reg = LinearRegression().fit(data.reshape(-1, 1), labels)
        labels_pred = line_reg.predict(data.reshape(-1, 1))
        r2=model_evaluation.r2_simple(list(labels), labels_pred.ravel())
        return r2, labels_pred, line_reg
    
    def _incremental_learn(A: list, B: list)->list:
    
        A_list=A.copy()
        B_list=B.copy()
        x=np.linspace(0, 1000)
        X=np.concatenate((x, x))
    
        while len(A_list)>1:

            f=[A_list.pop(0)+B_list.pop(0)*x, A_list.pop(0)+B_list.pop(0)*x]
            Y=np.concatenate((f[0], f[1]))
            line_reg = LinearRegression().fit(X.reshape(-1, 1), Y)
            a=line_reg.intercept_
            b=line_reg.coef_[0]
            A_list.insert(0, a)
            B_list.insert(0, b)
        return A_list, B_list
    
    def incremental_learn(A: list, B: list)->list:
        a=np.mean(A)
        b=np.mean(B)
        return a, b

class optim_fun:
    def __init__(self, A, min_p, B, eps, arg_shape):
        self.A=A
        self.min_p=min_p
        self.B=B
        self.eps=eps
        self.arg_shape=arg_shape
    def revenue(self, x):
        self.x=x
        rev = -1.*np.sum(self.x*((self.A*np.power(self.x-self.min_p+1., self.B)-1.)*(1.+self.eps)))
        return rev
    def f_con(self, x):
        self.x=x
        res_con=np.sum(np.array(((self.A*np.power(self.x-self.min_p+1., self.B)-1.)*(1.+self.eps))).reshape(self.arg_shape), axis=1)
        return res_con
