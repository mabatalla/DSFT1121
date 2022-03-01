import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import iqr
from IPython.display import display
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report

from wordcloud import WordCloud
from PIL import Image


# FUNCION 1
def visualizeME_palettes_or_colors(selection = 'palette', quantity_colors= 8):
    '''
    Function that returns the possible color palettes of the Seaborn library
    ### Parameters (2):
        * selection: `str` by default gives 'palettes', but you can choose colors
        * quantity_colors: `int` by default it returns 8 colors per palette, you can change number of colors you will need. And if you want to see colors, it is not neccesary this parameter.
    ### Return (1):
        * plt.show(): available palettes/ colors with their respective names
    '''
    colors = pd.read_csv('data/seaborn_color_list.csv')

    if selection == 'palette':
        grid = np.vstack((np.linspace(0, 1, quantity_colors), np.linspace(0, 1, quantity_colors)))
        options_colors = colors['PALETTE_COLORS'].dropna().sort_values(key=lambda x: x.str.lower())
        col = 4                             
        pos = 1 
        row = int(len(options_colors)/col)+1 
        plt.figure(figsize=(col*4,row))

        for i in options_colors:
            if '_r' in i:
                pass
            else:
                plt.subplot(row, col, pos)
                plt.imshow(grid, cmap = i, aspect='auto')
                plt.axis('off')
                plt.title(i, loc = 'center', fontsize = 20)
                pos = pos + 1
    
        print('If you want any palette reversed, just add "_r" at the end of the palette name')

    elif selection == 'color':
        just_colors = sorted(colors['CSS4_COLORS'].dropna().sort_values(key=lambda x: x.str.lower()))
        col = 4                             
        pos = 1 
        row = int(len(just_colors)/col)+1 
        plt.figure(figsize=(col*3,row))
        
        for i in just_colors:
            plt.subplot(row, col, pos)
            plt.hlines(0,0,5, color = i ,linestyles = 'solid', linewidth = 25)
            plt.axis('off')
            plt.text(0,0.04, i, fontsize = 20)
            pos = pos + 1

    plt.tight_layout()
    return plt.show()


# FUNCION 2
def visualizeME_and_describe_violinbox(dataframe, categ_var, numeric_var, palette= 'tab10', save= True):
    '''
    Function that allows to obtain a more complete graph by merging boxplot and violinplot together with a table of descriptive metrics
    ### Parameters (5):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` categoric variable
        * numeric_var:  `str` numeric variable
        * palette:  `str` by default 'tab10', but you can choose your palette
        * save:  `bool` by default True, the function save the plot and table generated
    '''
    # Generate ViolinBOX graph
    num_cat = len(list(dataframe[categ_var].unique()))
    plt.figure(figsize=(num_cat*1.5,10))
    sns.violinplot(x=categ_var, y=numeric_var, data=dataframe, palette= palette)
    ax = sns.boxplot(x=categ_var, y=numeric_var, data=dataframe,fliersize=0, color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var.upper() + '_vs_' + categ_var.upper()
    plt.title(titulo, fontsize=15);

    # Save graph
    if save == True:
        graph = 'visualizeME_Graphic_violinbox_' + titulo.lower() + '.png'
        plt.savefig(graph)

    # Metrics table
    cabeceras= ['Metrics',]
    fila1 = ['Upper limit',]
    fila2 = ['Q3',]
    fila3 = ['Median',]
    fila4 = ['Q1',]
    fila5 = ['Lower limit',]  
    iqr_ = iqr(dataframe[numeric_var], nan_policy='omit')
    d = [ fila1, fila2, fila3, fila4, fila5]
    for i in sorted(list(dataframe[categ_var].unique())):
        cabeceras.append(i)
        mediana = round(float(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].median()), 2)
        fila3.append(mediana)
        q1 = round(np.nanpercentile(dataframe[dataframe[categ_var].isin([i])][[numeric_var]], 25), 2)
        fila4.append(q1)
        q3 = round(np.nanpercentile(dataframe[dataframe[categ_var].isin([i])][[numeric_var]], 75), 2)
        fila2.append(q3)
        th1 = round(q1 - iqr_*1.5, 2)
        fila5.append(th1)
        th2 = round(q3 + iqr_*1.5, 2)
        fila1.append(th2)
    table = pd.DataFrame(d, columns=cabeceras)
    table = table.set_index('Metrics')
    
    # Save table
    if save == True:
        name = 'visualizeME_table_violinbox_' + titulo.lower() + '.csv'
        table.to_csv(name, header=True)
    
    plt.show()
    display(table)


# FUNCION 3
def visualizeME_and_describe_barplot(dataframe, categ_var, numeric_var, palette='tab10', save = True):
    '''
    Function that allows to obtain a barplot with a table of descriptive metrics
    ### Parameters (5):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` categoric variable
        * numeric_var:  `str` numeric variable
        * palette:  `str` by default 'tab10', but you can choose your palette
        * save:  `bool` by default True, the function save the plot and table generated
    '''
    # Graph
    num_cat = len(list(dataframe[categ_var].value_counts().index[::-1]))
    plt.figure(figsize=(num_cat*1.5,8))
    ax = sns.barplot(x= categ_var,y= numeric_var, data= dataframe, palette= palette, order= dataframe[categ_var].value_counts().index[::-1], ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var.upper() + ' vs. ' + categ_var.upper()
    plt.title(titulo, fontsize=15)
    
    # Save graph
    if save == True:
        path=os.path.join('visualizeME_Graphic_barplot_' + titulo.lower() + '.' + 'png')
        plt.savefig(path, format='png', dpi=300)
    
    # Metrics table
    eti_media= 'Mean ' + numeric_var
    eti_desv = 'Standard Deviation ' + numeric_var
    cabeceras= ['Metrics',]
    fila1 = ['Number of records',]
    fila2 = [eti_media,]
    fila3 = [eti_desv,]
    d = [fila1, fila2, fila3]
    for i in list(dataframe[categ_var].value_counts().index[::-1]):
        cabeceras.append(i)
        total = str(int(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].count()))
        fila1.append(total)
        media = round(float(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].mean()), 2)
        fila2.append(media)
        desv = round(float(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].std()), 2)
        fila3.append(desv)
    table = pd.DataFrame(d, columns=cabeceras)
    table = table.set_index('Metrics')
    
    # Save table
    if save == True:
        name = 'visualizeME_table_barplot_' + titulo.lower() + '.csv'
        table.to_csv(name, header=True)
    
    plt.show()
    display(table)


# FUNCION 4
def visualizeME_c_matrix(y_true, 
                        y_pred, 
                        title='',
                        categories=[],
                        rotate=False,
                        cmap='',
                        cbar=True,
                        metrics=True,
                        save=True):
    '''
    This function plots a prettier confusion matrix with useful annotations 
       
    and adds a pd.DataFrame with the most common metrics used in classification.
        
    You can use this function with binary or multiclass classification.
    
    The figsize is calculated based on the number of categories.
        
    For both cases:
        * In the main diagonal you get:
            * counts over total of class
            * percentaje over total observations
        * In the rest of the cells you get:
            * counts
            * percentaje over total observations
        * If counts is zero you get an empty cell
           
    In case the function founds binary catergories in your data then a binary
    
    matrix is displayed with the TN, FN, FP, TP tags.
         
    ### Parameters:
    
        y_true -- `array_like`
            True labels.

        y_pred -- `array_like`
            Predictions to compare with true labels.

        title='' -- `str`
            Title to be displayed and used to save file.

        categories=[] -- `list(str)`
            List of names of the classes to be displayed instead of numeric values.

        rotate=False -- `bool`
            Applies a rotation on xticklabels in case they overlap.

        cmap='' -- `Matplotlib colormap name or object, or list of colors` 
            If not provided default is 'Blues'.

        cbar=True -- `bool`
            Whether to draw a colorbar.
            
        metrics=True -- `bool`
            Displays a pd.DataFrame with the most common metrics.

        save=False -- `bool` 
            Saves plot and metrics (if metrics=True) to disk. If title='' default is 'classifier'.
        
    ### Return:
    
        cfm -- `sklearn.metrics.confusion_matrix`
        
        metrics -- `pd.DataFrame`
    '''
    sns.set_theme(style='whitegrid')                
    # Generate confusion matrix
    cfm = confusion_matrix(y_true, y_pred)
    
    # Set tipe of classifier: binary or not binary
    if len(cfm)==2:
        base = len(cfm)
        bin_classifier = True
        annot_kws = {"size": 10*base}
        title_fontsize = 12*base
        axis_fontsize = 10*base
        labels_fontsize = 10*base
        plt.figure(figsize=(4.5*base, 4.5*base))
        
    else:
        base = len(cfm)
        bin_classifier = False    
        annot_kws = {"size": 1.5*base}
        title_fontsize = 3.5*base
        axis_fontsize = 2.5*base
        labels_fontsize = 1.5*base
        plt.figure(figsize=(1.75*base, 1.75*base))
    
    # Calculate auxiliar data   
    cfm_rowsum  = np.sum(cfm, axis=1, keepdims=True)
    
    if bin_classifier:
        cfm_percent = cfm / np.sum(cfm).astype(float)
    else:
        cfm_percent = cfm / cfm_rowsum.astype(float)
    
    # Build empty matrix for labels
    labels = np.zeros_like(cfm).astype(str)
    
    # Iterate labels to write correct annot
    nrows, ncols = cfm.shape
    
    for i in range(nrows):
        for j in range(ncols):
            count = cfm[i, j]
            percent = cfm_percent[i, j]
            
            if i == j:
                sum = cfm_rowsum[i]
                labels[i, j] = f'{count} / {int(sum)}\n{percent:.2%}'
            elif count == 0:
                labels[i, j] = ''
            else:
                labels[i, j] = f'{count}\n{percent:.2%}'
    
    if bin_classifier:
        names = ['TN', 'FP', 'FN', 'TP']
        labels = [name + '\n' + label for name, label in zip(names, labels.flatten())]
        labels = np.asarray(labels).reshape(2, 2)
    
    # Set color map
    if cmap == '':
        cmap = 'Blues'
    else:
        pass
        
    # Generate heatmap
    ax = sns.heatmap(cfm,
                     annot=labels,
                     annot_kws=annot_kws,
                     fmt='',
                     square=True,
                     cmap=cmap,
                     cbar=cbar)
        
    # Define categories position
    cat_position = [(i + .5) for i in range(len(labels))]
    
    # Add label rotation        
    if rotate:
        xdegree=50
        plt.yticks(rotation=0)
    else:
        xdegree=0
        plt.yticks(rotation=0)
    
    # Set title label
    if title == '':
        title = 'TRUE VS. PREDICTED'
    else:
        pass
    
    ax.set_title(f'{title.upper()}', fontsize=title_fontsize)
    
    # Set axis labels
    ax.set_xlabel('PREDICTED LABEL', fontsize=axis_fontsize)
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('TRUE LABEL', fontsize=axis_fontsize)
    ax.yaxis.set_label_position('left')
    
    # Set tick labels
    # Define category names
    if categories != []:
        try:
            categories = [label for label in categories]
            ax.set_xticks(cat_position)
            ax.set_xticklabels(categories, fontsize=labels_fontsize, rotation=xdegree)
            ax.xaxis.tick_bottom()
            
            ax.set_yticklabels(categories, fontsize=labels_fontsize)
            ax.yaxis.tick_left()

        except ValueError:
            print('''Impossible to parse categories with number of classes. Ticklabels set to numeric.''')
            
    # Save plot
    if save:
        name = 'visualizeME_cf_matrix_' + title.lower() + '.png'
        path=os.path.join(name + '.' + 'png')
        plt.savefig(path, format='png', dpi=300)

    # Plot
    plt.show()
        
    # Calculate metrics
    if metrics:
        if bin_classifier:
            metrics_df = pd.DataFrame({title: [f'{accuracy_score(y_true, y_pred):.10f}',
                                            f'{precision_score(y_true, y_pred):.10f}',
                                            f'{recall_score(y_true, y_pred):.10f}',
                                            f'{f1_score(y_true, y_pred):.10f}',
                                            f'{roc_auc_score(y_true, y_pred):.10f}']},
                                   index=[['Accuracy: (TP + TN) / TOTAL',
                                           'Precision: TP / (TP + FP)',
                                           'Recall: TP / (TP + FN)',
                                           'F1: harmonic mean (accuracy, recall)',
                                           'ROC AUC']])
        else:
            report = classification_report(y_true, y_pred)
            report = [line.split(' ') for line in report.splitlines()]

            header = [x.upper() for x in report[0] if x!='']

            index = []
            values = []

            for row in report[1:-5]:
                row = [value for value in row if value!='']
                if row!=[]:
                    index.append(row[0].upper())
                    values.append(row[1:])

            index.append('ACCURACY')
            values.append(['-', '-'] + [x for x in report[-3] if x != ''][-2:])
            index.append('MACRO AVG.')
            values.append([x for x in report[-2] if x != ''][-4:])
            index.append('WEIGHTED AVG.')
            values.append([x for x in report[-1] if x != ''][-4:])

            metrics_df = pd.DataFrame(data=values, columns=header, index=index)

        # Plot metrics
        display(metrics_df)
    
    # Save metrics
    if save:
        name = 'visualizeME_cf_matrix_' + title.lower() + '.csv'
        metrics_df.to_csv(name, header=True)

    return cfm, metrics


# FUNCION 5
def visualizeME_FigureWords(dataframe, categ_var, shape= 'seahorse', cmap= 'tab10', contour= 'steelblue', back_color = 'white', height= 18, width = 20, save= True):
    '''
    Function that returns graph of words with different shapes, with the possibility to choose between 'dino', 'heart', 'star', 'seahorse' and 'hashtag'. I hope you like it!
    ### Parameters (9):
        * dataframe: `dataframe` origin table
        * categ_var: `str` categoric variable
        * shape: `str` by default is 'seahorse' shape, but you can choose from this list: 'seahorse', 'dino', 'heart', 'star' and 'hashtag'.
        * cmap: `str` by default is 'tab10', but you can choose your palette of Seaborn. If you want to know which palettes are available you can call visualizeME_colors_palettes() function
        * contour: `str` by default is 'steelblue', but you can choose your favourite color
        * back_color: `str` by default is 'white', but you can choose your background color
        * height: `int` by default is 18, but you can select your preference on height of the figure
        * width:`int` by default is 20, but you can select your preference on width of the figure
        * save: `bool` by default is True in order to save your graph, but if you prefer don't save it, just choose 'False'
    ### Return (1):
        * plt.show(): graph with your figure(by default will be seahorse)
    '''
    # Shape
    while shape not in ['dino', 'heart', 'star', 'seahorse', 'hashtag']:    
        shape = input('Try again, what shape do you want for your figure words graph?\n*Dino\n*Heart\n*Star\n*Seahorse: ').lower()
    if shape == 'seahorse':
        figure = 'data/seahorse_visualizeME.jpg'
    elif shape == 'dino':
        figure = 'data/dino_steg_visualizeME.jpg'
    elif shape == 'heart': 
        figure = 'data/corazon_visualizeME.png'
    elif shape == 'star':
        figure = 'data/estrella-silueta_visualizeME.png'  
    elif shape == 'hashtag':
        figure = 'data/hashtag-silueta_visualizeME.png'
    
    # Words
    words = ' '.join(map(str, dataframe[categ_var]))
    custom_mask = np.array(Image.open(figure))
    wordcloud = WordCloud(background_color=back_color,
                      width=2500,
                      height=2000,
                      max_words=500, 
                      contour_width=0.1, 
                      contour_color= contour, 
                      colormap= cmap,
                      scale =5,mask=custom_mask).generate(words)
    
    plt.figure(1, figsize = (height, width))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save Graph
    if save == True:
        figure = figure.split('/')[1]
        figure = figure.split('.')[0]
        name = 'visualizeME_Graphic_' + figure + '.png'
        plt.savefig(name)
    
    return plt.show()


# FUNCION 6
def visualizeME_bagel_look_top(dataframe, categ_var, top=0, cmap = 'tab10', circle=True, save=True):
    '''
    Function to generate a bagel graphic where you can select the top categories you want to see, or everyone by default
    ### Parameters (6):
        * dataframe: `dataframe` origin table
        * categ_var: `str` categoric variable
        * top: `int` by default is 0, but you can choose look the top n categories and their weights.
        * cmap: `str` by default is 'tab10', but you can choose your palette of Seaborn. If you want to know which palettes are available you can call visualizeME_colors_palettes() function
        * circle: `bool` by default is True, in orders to seems like a bagel, but you can choose select a pie
        * save: `bool` by default is True in order to save your graph, but if you prefer don't save it, just choose 'False'
    '''
    data_valores= dataframe[categ_var].value_counts()
    # Filter by top categories
    if top != 0:
        data_valores_top= data_valores[(top):]
        p=0
        for i in data_valores_top:
            p=i+p
        data_valores_top=pd.Series([p],index=[f"Resto [{len(data_valores_top.index)}]"])
        data_valores_max= data_valores[:(top)]
        data_valores=pd.concat([data_valores_max, data_valores_top])
    
    # Generate graph
    plt.figure(figsize=(10,10))
    plt.pie(data_valores.values, labels=data_valores.index, textprops={"fontsize":15}, startangle = 60, autopct='%1.2f%%', frame=False, colors= sns.color_palette(cmap))
    p=plt.gcf()
    if circle is True:
        my_circle=plt.Circle((0,0), 0.4, color="w")
        p.gca().add_artist(my_circle)
    titulo= 'DISTRIBUCIÓN DE ' + categ_var.upper()
    plt.title(titulo, fontsize= 20)

    # Save graph
    if save == True:
        path=os.path.join('visualizeME_Graphic_baggel_' + titulo.lower() + '.png')
        plt.savefig(path, format='png', dpi=300)
    
    # Generate table
    values_bagel = pd.DataFrame(dataframe[categ_var].value_counts())
    new_list = []
    for i in list(dataframe[categ_var].value_counts()):
        sumat = sum(list(dataframe[categ_var].value_counts()))
        peso = i/sumat
        new_list.append(peso)
    porcentaj_nums = list(map(lambda x : x * 100, new_list))
    porcentaj_round3 = list(map(lambda x : round(x,2), porcentaj_nums))
    porcentaj = list(map(lambda x : str(x) + '%', porcentaj_round3))
    values_bagel['Pesos(%)'] = porcentaj

    # Save table
    if save == True:
        name = 'visualizeME_table_bagel_' + titulo.lower() + '.csv'
        values_bagel.to_csv(name, header=True)
    
    plt.show()
    return display(values_bagel)
    