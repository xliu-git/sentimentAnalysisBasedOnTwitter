'''
Code of data analyzation part:
Compute polarity at first then randomize the dataset. Analyze the balanced dataset and convert the anlysis into figures.

Input:
1. In Main function, change the Names list to the corresponding Names list in dataCollection.py

Compile:
Run 'python dataCollect.py'

Output:
1. Histogram figrues of each candidate
2. Boxplot figures of each candidate
3. Displot figures of each candidate
4. 5 Most negative comments of each candidate
5. 5 Most positive comments of each candidate
6. Word cloud figure of each candidate
7. Postive pie chart
8. Negative pie chart
9. Comparision Barplot 
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

'''
Function ComputeColumnPolarityAndExpression:
This function aims at data preparation. Firslty creating a data frame by the read functions of pandas. Then set up column polarity by applying helper function ComputePolarity(text). Next set up Expression Label by applying helper function ComputeExpression(text). Calling already built-in functions in Textblob. Finally return the data set with Polairty and Expression Label.

Input:
1. Function ComputeColumnAndExpression: CSV file
2. Helper function ComputePolarity: string to be computed its polarity
3. Helper function ComputeExpression: int to be classified to "Positive", "Negative" or "Neutral"

Output:
Dataset of pandas data frame with polarity and expression label of each comment  
'''
def ComputePolarity(text):
    return TextBlob(text).sentiment.polarity

def ComputeExpression(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

def ComputeColumnPolarityAndExpression(CsvFile):
    dataset = pd.read_csv(CsvFile, encoding='utf-8')
    
    dataset['Sentiment_Polarity'] = dataset['text'].apply(ComputePolarity)

    dataset['Expression_Label'] = dataset['Sentiment_Polarity'].apply(ComputeExpression)

    return dataset

'''
Function HistogramFigure:
Creating histogram of each candidates comments with different expression label.

Input:
1. prepared dataset
2. figure name string

Output:
Histogram figure of polarity distrubution in comments
'''
def HistogramFigure(dataset, figureName):
    datasetClassified = dataset.groupby('Expression_Label').count()
    
    figX = list(datasetClassified['Sentiment_Polarity'])
    figY = list(datasetClassified.index)
    df = pd.DataFrame(list(zip(figX, figY)), columns=['x','y'])
    df['color'] = pd.Series(['red', 'blue', 'green'])

    fig = go.Figure(go.Bar(x=df['x'], y=df['y'], orientation='h', marker={'color': df['color']}))
    fig.update_layout(title='{} Polarity Histogram'.format(figureName))
    fig.write_image("{} Histogram.png".format(figureName))

'''
Function SetDataSize:
SetDataSize function computes the suitable size for further processing without neutral comments

Input:
Two prepasred dataset of candidates

Output:
Int number of suitable size which equals to the minimal(negative + positive)-50  
'''
def SetDataSize(dataset1, dataset2):
    dataset1Classified = dataset1.groupby('Expression_Label').count()
    dataset2Classified = dataset2.groupby('Expression_Label').count()
    
    datasize = min(
        dataset1Classified.loc['Negative', 'Sentiment_Polarity']+dataset1Classified.loc['Positive', 'Sentiment_Polarity'],
        dataset2Classified.loc['Negative', 'Sentiment_Polarity']+dataset2Classified.loc['Positive', 'Sentiment_Polarity']
    )
    return datasize-50

'''
Function DropNeutralAndBlananceData:
To further analyze the data, this function drop all neurtal comments and randomized rest comments in a fixed size

Input:
1. Dataset of candidate
2. Fixed data size after randomization

Output:
Balanced fixed size data size 
'''
def DropNeutralAndBalanceData(dataset, size):
    dataset = dataset[~dataset['Expression_Label'].isin(['Neutral'])]
    
    np.random.seed(10)
    remove_n =dataset.shape[0]-size
    drop_indices = np.random.choice(dataset.index, remove_n, replace=False)
    balanced_dataset = dataset.drop(drop_indices)
    
    return balanced_dataset

'''
Function DisplotFigure:
Create displot figure of each candidate

Input:
1. Balanced dataset
2. String figure name

Output:
displot figure named figureName.png
'''
def DisplotFigure(balanced_dataset, figureName):
    displotFig = sns.distplot(balanced_dataset['Sentiment_Polarity'])
    fig = displotFig.get_figure()
    fig.savefig('{} displot.png'.format(figureName))
    plt.close(fig)

'''
Function BoxplotFigure:
Create boxplot figure of each candidate.

Input:
1. Blanced dataset
2. String figure name

Output:
Boxplot figure name figureName.png 
'''
def BoxplotFigure(balanced_dataset, figureName):
    boxplotFig = sns.boxplot(balanced_dataset['Sentiment_Polarity'])
    fig = boxplotFig.get_figure()
    fig.savefig('{} boxplot.png'.format(figureName))
    plt.close(fig)

'''
Function BarplotAndPieFigure:
Create barplot and pir figure for these two candidates with helper function BarplotHelper. The BarplotHelper function computes the nagative and postive count of each candidate.

Input:
1. Both balanced candidates' dataset
2. String figure name

Output:
1. Comparision barplot figure of two candidates
2. Negative pie chart of two candidates
3. Positive pie chart of two candidates
'''
def BarplotHelper(balanced_dataset):
    totalCount = balanced_dataset.groupby('Expression_Label').count()
    negativeCount = totalCount['user'][0]
    positiveCount = totalCount['user'][1]
    return negativeCount, positiveCount

def BarplotAndPieFigure(balanced_dataset1, balanced_dataset2, figureNames):
    neg1, pos1=BarplotHelper(balanced_dataset1)
    neg2, pos2=BarplotHelper(balanced_dataset2)

    BarPos = [pos1, pos2]
    BarNeg = [neg1, neg2]

    fig = go.Figure(data=[go.Bar(name='Positive', x=figureNames, y=BarPos), go.Bar(name='Negative', x=figureNames, y=BarNeg)])
    fig.update_layout(barmode='group')
    fig.write_image("Comparision Barplot.png")

    labelNeg = ['Negative_{}'.format(figureNames[0]), 'Negative_{}'.format(figureNames[1])]
    labelPos = ['Positive_{}'.format(figureNames[0]), 'Positive_{}'.format(figureNames[1])]
    explode = (0.1, 0.1)

    fig1, PieNeg = plt.subplots()
    PieNeg.pie(BarNeg, explode=explode, labels = labelNeg, autopct = '%1.1f%%', shadow = True, startangle=90)
    PieNeg.set_title('Negative Pie')
    PieNeg.get_figure().savefig('negative pie chart.png')

    fig2, PiePos = plt.subplots()
    PiePos.pie(BarPos, explode=explode, labels = labelPos, autopct = '%1.1f%%', shadow = True, startangle=90)
    PiePos.set_title('Positive Pie')
    PiePos.get_figure().savefig('positive pie chart.png')

'''
Function MostPosAndNegComments:
Return 5 most negative and postive comments of candidate.

Input:
1. Balanced dataset
2. String figure name

Output:
1. 5 most negative comments figure
2. 5 most positive comments figure
'''
def MostPosAndNegComments(balanced_dataset, figureName):
    balanced_dataset.sort_values(by=['Sentiment_Polarity'], inplace=True)
    
    negative_texts = balanced_dataset['text'].head()
    negative_texts = list(negative_texts)
    negative_polarities = balanced_dataset['Sentiment_Polarity'].head()
    negative_polarities = list(negative_polarities)

    fig = go.Figure(data=[go.Table(header=dict(values=['Most Positive Replies of {}'.format(figureName), 'Polarity'], fill_color='paleturquoise', align='left'), cells=dict(values=[negative_texts, negative_polarities], fill_color='lavender', align='left'))])
    fig.write_image("most negative replies of {}.png".format(figureName))

    positive_texts = balanced_dataset['text'].tail()
    positive_texts = list(positive_texts)
    positive_polarities = balanced_dataset['Sentiment_Polarity'].tail()
    positive_polarities = list(positive_polarities)

    fig = go.Figure(data=[go.Table(header=dict(values=['Most Positive Replies of {}'.format(figureName),'Polarity'], fill_color='paleturquoise', align='left'), cells=dict(values=[positive_texts, positive_polarities], fill_color='lavender', align='left'))])
    fig.write_image("most positive replies of {}.png".format(figureName))

'''
Function WordCloudFigure:
Create word cloud by most frequent words in the comments for each candidate

Input:
1. Balanced dataset
2. String figure name 

Output:
Word cloud figure
'''
def WordCloudFigures(balanced_dataset, figureName):
    text = str(balanced_dataset['text'])
    wordcloud = WordCloud(max_font_size=100, max_words=500, background_color="white", colormap = "rainbow").generate(text)
    wordcloud.to_file("{} word cloud.png".format(figureName))    
        
'''
Function Main:
Start the whole process of data analyzing

Input:
Change the Names list as corresponding Names list in dataCollect.py
'''
if __name__ == '__main__':

    Names = ['Pence', 'Harris']
    balanced_dataset_list = []
    dataset_list = []
    
    for name in Names:
        dataset = ComputeColumnPolarityAndExpression('{}_data.csv'.format(name))
        dataset_list.append(dataset)
        HistogramFigure(dataset, name)

    datasize = SetDataSize(dataset_list[0], dataset_list[1])

    for i in range(2):
        balanced_dataset = DropNeutralAndBalanceData(dataset_list[i], datasize)
        balanced_dataset_list.append(balanced_dataset)

        WordCloudFigures(balanced_dataset, Names[i])
        DisplotFigure(balanced_dataset, Names[i])
        BoxplotFigure(balanced_dataset, Names[i])
        MostPosAndNegComments(balanced_dataset, Names[i])
    

    BarplotAndPieFigure(balanced_dataset_list[0], balanced_dataset_list[1], Names)    
    
    
