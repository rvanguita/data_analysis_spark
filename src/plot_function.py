import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import shap
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    log_loss, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
)

from sklearn.metrics import confusion_matrix

class ShapPlot:
    def __init__(self, model, X_test, df_feature, features_drop=None):
        self.model = model
        self.X_test = X_test
        
        self.feature_names = df_feature.columns
        if features_drop:
            self.feature_names = df_feature.drop(features_drop, axis=1).columns
            
        self.explainer = shap.Explainer(self.model)  # Reutiliza o explainer para consistência
        
    def first_analysis(self, extension=False):
        shap_values = self.explainer(self.X_test)

        # Ajuste os nomes das features
        shap_values.feature_names = self.feature_names

        # Gráficos principais
        shap.plots.beeswarm(shap_values)
        shap.plots.bar(shap_values)
        shap.waterfall_plot(shap_values[0])

        # Extensão opcional para análise adicional
        if extension:
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)
        
    def complete(self, comparison=False, analysis=None, interaction_index=None, show_interactions=False):
        # Calcule os valores SHAP
        shap_values = self.explainer(self.X_test)

        # Converter X_test para DataFrame para melhores visualizações
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        # Criar o objeto Explanation para o force_plot
        shap_explanation = shap.Explanation(
            values=shap_values.values[0],  # Os valores SHAP
            base_values=self.explainer.expected_value,  # O valor esperado (média das previsões)
            data=X_test_df.iloc[0],  # A amostra de entrada que estamos explicando
            feature_names=self.feature_names  # Os nomes das features
        )

        # Exibir força de explicação para uma previsão
        shap.initjs()
        shap.force_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)

        # Criar gráfico de decisão
        shap.decision_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)

        # Calcular e exibir interações SHAP, se necessário
        if show_interactions:
            shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)  # Correção para `self.explainer`
            shap.summary_plot(shap_interaction_values, self.X_test, feature_names=self.feature_names)

        # Se `comparison` for verdadeiro, criar gráfico de dependência
        if comparison and analysis is not None:
            shap.dependence_plot(analysis, shap_values, self.X_test, feature_names=self.feature_names, interaction_index=interaction_index)


class ClassificationHyperTuner:
    def __init__(self, X_train, y_train, X_test, y_test, n_trials=30, model_name=None):
        # Initialize data and configuration
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.model_name = model_name

    def objective(self, trial):
        # Select model based on the given model name
        if self.model_name == "cat":
            return self.boost_cat(trial)
        elif self.model_name == "lgb":
            return self.boost_lgb(trial)
        elif self.model_name == "xgb":
            return self.boost_xgb(trial)
        else:
            raise ValueError("Unsupported model. Choose between 'cat', 'lgb', or 'xgb'.")


    def run_optimization(self):
        # Study for hyperparameter optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best AUC: {study.best_value}")
        
        return study.best_params, study.best_value
    
    
    def train_and_evaluate(self, model):
        # Fit the model and evaluate it on the test set
        model.fit(self.X_train, 
                  self.y_train, 
                #   eval_set=[(self.X_test, self.y_test)],
                #   early_stopping_rounds=100, 
                  verbose=False)
        
        predictions = model.predict_proba(self.X_test)[:, 1]
        
        
        accuracy = accuracy_score(self.y_test, predictions > 0.5)
        # auc = roc_auc_score(self.y_test, predictions)
        return accuracy


    def boost_cat(self, trial):
        # Define hyperparameters for CatBoost
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 5000),  # Aumentar o número máximo de iterações
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.2, log=True),  # Ajuste mais fino para LR
            "depth": trial.suggest_int("depth", 4, 12),  # Foco em valores de profundidade intermediários
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),  # Evitar extremos como 0.5 ou 1.0
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),  # Uso de mais recursos por nível
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 100),  # Ajustado para dados mais densos
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10.0, log=True),  # Regularização
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),  # Evitar o Depthwise
            "border_count": trial.suggest_int("border_count", 50, 150),  # Limite menor de contagens de borda
            "od_type": trial.suggest_categorical("od_type", ["Iter", "IncToDec"]),
        }

        model = cb.CatBoostClassifier(**params, silent=True)
        return self.train_and_evaluate(model)


    def boost_lgb(self, trial):
        # Define hyperparameters for LightGBM
        params = {
            "objective": "binary",
            "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),  # Aumentado o range de estimadores
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),  # Ajustado para menor LR
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),  # Valor máximo ajustado para 255, típico para LightGBM
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),  # Mais conservador na subsample
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),  # Similarmente ajustado
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),  # Limites ajustados para dados maiores
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),  # Regularização L1
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),  # Regularização L2
            "max_bin": trial.suggest_int("max_bin", 100, 300),  # Ajustado para bins maiores
        }

        model = lgb.LGBMClassifier(**params)
        return self.train_and_evaluate(model)


    def boost_xgb(self, trial):
        # Define hyperparameters for XGBoost
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        # params = {
        #     'objective': 'binary:logistic',
        #     'n_estimators': trial.suggest_int('n_estimators', 100, 2500),  # Maior número de estimadores
        #     'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),  # LR menor para explorações mais estáveis
        #     'max_depth': trial.suggest_int('max_depth', 4, 10),  # Faixa maior para a profundidade máxima
        #     'subsample': trial.suggest_float('subsample', 0.6, 0.85),  # Faixa otimizada de subsample
        #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  # Evitar valores muito extremos
        #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  # Peso mínimo mais amplo
        #     'gamma': trial.suggest_float('gamma', 0, 5.0),  # Ajuste para maior controle de regularização
        #     'lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True),  # Regularização L2 (lambda)
        #     'alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True),  # Regularização L1 (alpha)
        # }

        model = xgb.XGBClassifier(**params)
        return self.train_and_evaluate(model)


class TrainingValidation:
    def __init__(self, model, rouc_curve=False, confusion_matrix=False):
        self.rouc_curve = rouc_curve
        self.model = model
        self.confusion_matrix = confusion_matrix


    def plot_roc_curve(self, fpr, tpr, roc_auc, figsize=(4, 3), color='#c53b53'):
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color=color, lw=4, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linha diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve', fontsize=16, weight='bold')
        plt.legend(loc="lower right")
        

        plt.gca().grid(False)
        plt.gca().yaxis.set_visible(True)
        plt.gca().xaxis.set_visible(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.show()


    def calculate_metrics(self, y, predictions, predictions_proba):
        metrics = {
            'Accuracy': accuracy_score(y, predictions),
            'Precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'Recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'F1 Score': f1_score(y, predictions, average='weighted', zero_division=0),
            'ROC AUC': roc_auc_score(y, predictions_proba),
            'Matthews Corrcoef': matthews_corrcoef(y, predictions),
            'Cohen Kappa': cohen_kappa_score(y, predictions),
            'Log Loss': log_loss(y, predictions_proba)
        }
        return {k: round(v * 100, 2) if k != 'Matthews Corrcoef' and k != 'Cohen Kappa' else round(v, 2) 
                for k, v in metrics.items()}


    def plot_confusion_matrix(self, y, predictions):
        cm = confusion_matrix(y, predictions)
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
                for item in cm.flatten()
            ]
        ).reshape(2, 2)

        # Plotting the confusion matrix.
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=labels, fmt="")
        plt.title('Confusion Matrix', fontsize=16, weight='bold')
        plt.ylabel("True")
        plt.xlabel("Predicted")
        

    def normal(self, X, y, oversampling=False):
        self.model.fit(X, y)
        predictions_proba = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        
        scores = self.calculate_metrics(y, predictions, predictions_proba)
        scores_df = pd.DataFrame([scores])
        if oversampling:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)        
        if self.confusion_matrix:
            # print("Confusion Matrix:\n", confusion_matrix(y, predictions))
            self.plot_confusion_matrix(y, predictions)
        
        if self.rouc_curve:

            fpr, tpr, _ = roc_curve(y, predictions_proba)
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)

        return scores_df


    def cross(self, X, y, n_splits=5, oversampling=False):
        cv = KFold(n_splits=n_splits, shuffle=True)
        metrics_cross = {key: [] for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 
                                             'Matthews Corrcoef', 'Cohen Kappa', 'Log Loss']}
        
        all_predictions = []
        all_true_labels = []
        fpr_list = []
        tpr_list = []
        roc_auc_scores = []

        for idx_train, idx_test in cv.split(X, y):
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

            if oversampling:
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            predict_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = self.calculate_metrics(y_test, predictions, predict_proba)
            
            for key in metrics_cross.keys():
                metrics_cross[key].append(metrics[key])


            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)
                
                
            if self.rouc_curve:    
                fpr, tpr, _ = roc_curve(y_test, predict_proba)
                roc_auc = auc(fpr, tpr)
                roc_auc_scores.append(roc_auc)
                fpr_list.append(fpr)
                tpr_list.append(tpr)

        if self.confusion_matrix:
            self.plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions))


        if self.rouc_curve:
            self.plot_roc_curve(fpr_list[-1], tpr_list[-1], roc_auc_scores[-1])



        scores = {key: round(np.mean(val), 2) for key, val in metrics_cross.items()}
        # scores = {key: round(np.mean(val), 2) if key != 'Matthews Corrcoef' and key != 'Cohen Kappa' 
        #           else round(np.mean(val), 2) for key, val in metrics_cross.items()}
        scores_df = pd.DataFrame([scores])
        
        return scores_df


class DataVisualizer:
    def __init__(self, df, color='#c3e88d', figsize=(24, 12)):
        """
        Initializes the DataVisualizer with a DataFrame and default plot settings.

        Args:
        - df (pd.DataFrame): DataFrame containing the data.
        - color (str, optional): Default color for the plots.
        - figsize (tuple, optional): Default figure size.
        """
        self.df = df
        self.color = color
        self.figsize = figsize

    def plot_barplot(self, features, hue=None, custom_palette=None):
        rows, cols = self._calculate_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            grouped = self.df.groupby([feature, hue]).size().reset_index(name='count')
            total_count = grouped['count'].sum()
            grouped['percentage'] = (grouped['count'] / total_count) * 100

            num_categories = self.df[feature].nunique()
            width = 0.8 if num_categories <= 5 else 0.6

            sns.barplot(data=grouped, x='count', y=feature, palette=custom_palette, hue=hue, ax=ax, width=width, orient='h')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._customize_ax(ax)
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(True)
            ax.xaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
            for p in ax.patches:
                width = p.get_width()
                percentage = (width / total_count) * 100
                if percentage != 0:
                    ax.annotate(f'{percentage:.1f}%', 
                                (width, p.get_y() + p.get_height() / 2),
                                xytext=(5, 0), 
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=11, color='black', fontweight='bold')


        self._remove_extra_axes(axes, len(features))
        plt.tight_layout()

    def plot_boxplot(self, features, hue=None, custom_palette=None):
        rows, cols = self._calculate_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            if hue:
                sns.boxplot(data=self.df, x=feature, y=hue, hue=hue, orient='h', palette=custom_palette, ax=ax)
                ax.set_ylabel('')
            else:
                sns.boxplot(data=self.df, x=feature, color=self.color, orient='h', ax=ax)
                ax.yaxis.set_visible(False)
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._customize_ax(ax)

            ax.set_xlabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)

        self._remove_extra_axes(axes, len(features))
        plt.tight_layout()

    def plot_histplot(self, features, hue=None, custom_palette=None, kde=False):
        rows, cols = self._calculate_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            sns.histplot(data=self.df, x=feature, hue=hue, palette=custom_palette, kde=kde, ax=ax, stat='proportion')

            self._customize_ax(ax)

            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            

            # ax.legend(loc='upper right')

            ax.set_xlabel('')
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
        self._remove_extra_axes(axes, len(features))
        plt.tight_layout()

    def plot_seaborn_bar(self, hue, custom_palette):
        """
        Plots a bar chart displaying the percentage distribution of a categorical variable.
        """
        percentage_exited = self.df[hue].value_counts(normalize=True) * 100
        percentage_df = percentage_exited.reset_index()
        percentage_df.columns = [hue, 'percentage']

        plt.figure(figsize=(4, 3))
        ax = sns.barplot(data=percentage_df, x=hue, y='percentage', hue=hue, palette=custom_palette)
        
        ax.set_title(f'{percentage_exited.idxmin()} rate is about {percentage_exited.min():.2f}%', 
                     fontweight='bold', fontsize=13, pad=15, loc='center')
        
        ax.set_xlabel('')
        ax.invert_xaxis()
        ax.tick_params(axis='both', which='both', length=0)
        ax.yaxis.set_visible(False)
        self._customize_ax(ax, hide_spines=True)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}%', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, -10), 
                        textcoords='offset points',
                        fontsize=11, color='white', fontweight='bold')

        plt.tight_layout()

    def plot_custom_scatterplot(self, x, y, hue, custom_palette, title_fontsize=16, label_fontsize=14):
        """
        Plots a customized scatter plot with specified x and y axes, hue for color coding, and custom palette.
        """
        plt.figure(figsize=(16, 6))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, palette=custom_palette, alpha=0.6)

        plt.title(f'{x} vs {y}', fontsize=title_fontsize, weight='bold')
        plt.xlabel(x, fontsize=label_fontsize)
        plt.ylabel(y, fontsize=label_fontsize)

        ax = plt.gca()
        self._customize_ax(ax, hide_spines=True)

        legend = ax.get_legend()
        if legend:
            legend.set_title(hue)
            legend.set_bbox_to_anchor((1.15, 0.8))

    def _calculate_grid(self, num_features):
        rows = (num_features + 2) // 3
        cols = min(3, num_features)
        return rows, cols

    def _customize_ax(self, ax, hide_spines=False):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if hide_spines:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        ax.grid(False)

    def _remove_extra_axes(self, axes, num_features):
        for j in range(num_features, len(axes)):
            plt.delaxes(axes[j])
            
            
            
