import sys
import os
from os import walk
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas_profiling
from pandas_profiling import ProfileReport
from tabulate import tabulate
from IPython.display import display, HTML
import statistics 
from statistics import mode
from statistics import StatisticsError
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK']='True' #This fixes a bug in XGB that crashes the Kernel
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
import itertools
from sklearn.metrics import confusion_matrix
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers

class Classy:
    def __init__(self):
        self.chosen_file = None
        self.og_df = None
        self.y_name = None
        self.x = None
        self.y = None
        self.model_comparison = {}
        self.scrubbed = False
        self.best_model = None
        self.scaler = None
        print('\n––––––––––––––––––––––––––––––––––––')
        print(self.color.BOLD+'Welcome to CLASSY!'+self.color.END)
        print('––––––––––––––––––––––––––––––––––––\n')
        print('Call the ' + self.color.BLUE +'about()'+self.color.END + ' function to learn more about how CLASSY works.')
        print('OR Call the ' + self.color.BLUE +'obtain()'+self.color.END + ' function to select your data and get started.\n')
    
    def about(self):
        print('------------------------------------')
        print('About ClassifyBot')
        print('------------------------------------')
        
        print('\nCLASSY allows anybody to setup a basic ML classification workflow in less than 10 lines of code, with any dataset, and minimal prior experience required.')
        print('CLASSY makes both binary and multi-class classification simple. The full source code is available and can be customized to suit your needs.')
        print('CLASSY has 5 primary functions (recommended to call in sequential order):')
        print('\t'+self.color.BLUE+'–obtain()'+self.color.END)
        print('\t'+self.color.BLUE+'–explore()'+self.color.END)
        print('\t'+self.color.BLUE+'–scrub()'+self.color.END)
        print('\t'+self.color.BLUE+'–model()'+self.color.END)
        print('\t'+self.color.BLUE+'–interpret()'+self.color.END)
        print('\t'+self.color.BLUE+'–predict()'+self.color.END)
        print('\nCall the ' + self.color.BLUE +'obtain()'+self.color.END +' function to select your data and get started.\n')
    
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
    
    #Reusable confusion matrix function
    def conf_mat(self,name,val_preds,y_test):

        cnf_matrix = confusion_matrix(val_preds, y_test)
        plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) #Create the basic matrix.

        #Add title and Axis Labels
        plt.title('Confusion Matrix for ' + str(name))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        #Add appropriate Axis Scales
        
        class_names = np.unique(y_test) #Get class labels to add to matrix
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)


        #Add Labels to Each Cell
        thresh = cnf_matrix.max() / 2. #Used for text coloring below
        #Here we iterate through the confusion matrix and append labels to our visualization.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
                plt.text(j, i, cnf_matrix[i, j],
                         horizontalalignment="center",
                         color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.colorbar()
        plt.show()
        
    
    #Generic Error Message Function
    def error(self):
        print(self.color.RED +'\nWhoops, something went wrong. Please try launching Classify-Bot again.'+ self.color.END)
        
    #Obtain Data Function
    def obtain(self):
        print('\n------------------------------------')
        print('Obtaining Your Data')
        print('------------------------------------')
        
        self.scrubbed = False
        
        print(self.color.BOLD+"\nSelect The File You Want to Work With\n"+self.color.END)
        mypath = os.path.abspath(os.getcwd())
        files = []
        for (dirpath, dirnames, filenames) in walk(mypath):
            files.extend(filenames)
            break
        csvs = []
        for file in files:
            if '.csv' in file:
                csvs.append(file)
        print('I found ' + self.color.BLUE + self.color.BOLD + str(len(csvs)) + self.color.END +' CSV files in your directory that we can get started with.')
        if len(csvs) > 0:
            print('Which data set do you want to work with?\n')
            i = 1
            for csv in csvs:
                print(i,'-',csv)
                i+=1
            print(self.color.GREEN + '\nEnter the number of the file'+ self.color.END)

            file_choice = input()

            if file_choice.isnumeric() and (int(file_choice) in range(1,len(csvs)+1)):
                self.chosen_file = csvs[int(file_choice)-1]
                self.og_df = pd.read_csv(self.chosen_file)
                
                #Identify Target –––––––––––––––––––––––––
                print(self.color.BOLD+"\nIdentify Target\n"+self.color.END)    
                i = 1 #counter
                col_array = []
                for col in self.og_df.columns: 
                    print(self.color.BOLD,str(i) +':',col,self.color.END, '(DType:',self.og_df.dtypes[col],  '– NUnique:', self.og_df[col].nunique(),')')
                    i+=1
                    col_array.append(col)
                target = input("Which of these columns represents the category you are trying to predict?" + self.color.GREEN + ' Type in the associated column number below.'+ self.color.END+'\n')        
                if target.isnumeric() and (int(target) in range(1,len(self.og_df.columns)+1)):
                    target = int(target)
                    y_col = col_array[target-1]

                    self.y_name = y_col

                    print('\nTarget Variable Selected =',self.color.BLUE+str(y_col)+self.color.END)
                else:
                    return None 
                
                
                self.y = self.og_df[self.y_name]
                self.x = self.og_df.drop(self.y_name,axis=1)
                
                #Instructions
                print(self.color.BOLD+"\nNext Steps\n"+self.color.END)
                print('\n'+'Great. All set to get started with ' + self.chosen_file + '. Call the' + self.color.BLUE + ' scrub()' + self.color.END + ' function to continue.')
                print('\nYou can also explore the unscrubbed dataset by calling the ' + self.color.BLUE + 'explore()' + self.color.END + ' function.')
                print('\nLastly, If your file is already scrubbed and model ready, you can proceed straight to the ' + self.color.BLUE + ' model()' + self.color.END + ' function.')
            else:
                self.error()
                return(None)
        else:
            print('Please upload a CSV File to Begin')
            return(None)
        
 
    #Scrubbing Function
    def scrub(self):   
        print('\n------------------------------------')
        print('Scrubbing Your Data')
        print('------------------------------------')
        df = pd.read_csv(self.chosen_file)
        self.x = pd.read_csv(self.chosen_file)
        col = df.shape[1]
        row = df.shape[0]
        print('\n' + self.chosen_file + ' has ' + str(col) + ' columns and ' + str(row) + ' rows\n')
        
        #Take a smaller sample –––––––––––––––––––––––––
        print(self.color.BOLD+"Step 0. Take a sample (optional)\n"+self.color.END)
        if len(df) > 100000:
            print("Looks like your working with a LOT of data! That's great, but...")
            print("Sometimes that can make our models take a long time to run on a regular computer.")
            choice = input("Would you like to take a smaller sample of the data to speed things up?" + self.color.GREEN + ' Type (y/n) below'+ self.color.END+'\n')
            if choice == 'y':
                amount = input("What sized sample? (I recommend starting with ~50K rows)" + self.color.GREEN + ' Type a number below.'+ self.color.END+'\n')
                if amount.isnumeric():
                    amount = int(amount)
                    self.x = self.x.sample(n=amount,random_state=0)
                    df = df.sample(n=amount,random_state=0)
                else:
                    self.error()
                    return None
            elif choice != 'n':
                self.error()
                return None
        else:
            print("Skipping this step (since you're dataset isn't overwhelmingly large)")
        
        print(self.color.BOLD+"Step 1. Remove any target values (optional)\n"+self.color.END)
        #FUNCTION FOR REMOVING SPECIFIC TARGET VALUES FROM THE CLASSIFIER.
        print('\nYour target variable has '+ self.color.BLUE + str(self.x[self.y_name].nunique()) + self.color.END + ' unique values:')
        print('Value counts in ' + self.color.RED + 'red' + self.color.END + ' represent less than 5% of total data (consider removing).')
        
        val_array = []
        
        for i in range(len(self.x[self.y_name].unique())):
            if int(list(self.x[self.y_name].value_counts(dropna=False))[i]) < (.05*len(self.x[self.y_name])):  
                print(self.color.BOLD + str(i+1) + ': ' + str(list(self.x[self.y_name].value_counts(dropna=False).index)[i]) + self.color.END, self.color.RED, str(list(self.x[self.y_name].value_counts(dropna=False))[i]) + self.color.END)
            else: 
                print(self.color.BOLD + str(i+1) + ': ' + str(list(self.x[self.y_name].value_counts(dropna=False).index)[i]) + self.color.END, self.x[self.y_name].value_counts(dropna=False)[i])
            val_array.append(self.x[self.y_name].unique()[i])
            
        print("Would you like to remove of any of these values (and the associated rows)?")
        yorno = input("Don't worry about NAN values for now... we'll deal with that later" + self.color.GREEN + ' Type (y/n) below'+ self.color.END+'\n')
        
        if yorno == 'y':
            val_to_remove = input(self.color.GREEN + 'Please type in the number of each target value you want to remove, sepparating each number by a space.'+ self.color.END+'\n')
            val_to_remove = val_to_remove.split()
            val_to_remove = list(map(int, val_to_remove))
            
            for val in val_to_remove:
                print(val)
                remove = val_array[val-1]
                self.x[self.y_name] = self.x[self.y_name][self.x[self.y_name] != remove]
                
        
        elif yorno != 'n':
            self.error()
            return None
        
        print(self.x[self.y_name].value_counts(dropna=False))

        
        #Removing Columns –––––––––––––––––––––––––
        print(self.color.BOLD+"\nStep 2. Remove Columns (optional)\n"+self.color.END)        
        
        i = 1 #counter
        col_array = []
        for col in self.x.columns: 
            print(str(i) +':',col)
            i+=1
            col_array.append(col)
        
        yorno = input('\nWould you like to remove any of the columns?'+ self.color.GREEN + ' Type (y/n) below'+ self.color.END+'\n')
        if yorno == 'y':
            col_to_remove = input(self.color.GREEN + 'Please type in the number of each column you want to remove, sepparating each number by a space.'+ self.color.END+'\n')
            col_to_remove = col_to_remove.split()
            col_to_remove = list(map(int, col_to_remove))

            for col in col_to_remove:
                remove = col_array[col-1]
                self.x = self.x.drop(remove,axis=1)

            print(str(len(col_to_remove)) + ' columns were successfully removed.')
        elif yorno != 'n':
            self.error()
            return None
        

        #Fixing Any DataTypes –––––––––––––––––––––––––
        print(self.color.BOLD+"\nStep 3. Fixing Column Data Types (optional) \n"+self.color.END)        
        
        i = 1 #counter
        col_array = []
        for col in self.x.columns: 
            print(str(i) +':',col,'– DType:',self.x.dtypes[col],'– Examples (' + str(self.x[col].iloc[1]) +', ' + str(self.x[col].iloc[2]) + ', ' + str(self.x[col].iloc[3]) + ')')
            i+=1
            col_array.append(col)
        
        yorno = input('\nDo any of these column datatypes look incorrect?'+ self.color.GREEN + ' Type (y/n) below'+ self.color.END+'\n')
        if yorno == 'y':
            col_to_fix = input(self.color.GREEN + 'Please type in the number of each column you want to switch the datatype for, sepparating each number by a space.'+ self.color.END+'\n')
            col_to_fix = col_to_fix.split()
            col_to_fix = list(map(int, col_to_fix))

            for col in col_to_fix:
                fix = col_array[col-1]
                
                if(self.x[fix].dtype == np.float64 or self.x[fix].dtype == np.int64):
                    self.x[fix] = self.x[fix].astype(str)
                
                else: 
                    self.x[fix] = pd.to_numeric(self.x[fix], errors='coerce')
                    
            print(str(len(col_to_fix)) + ' column data types were successfully changed.')
        elif yorno != 'n':
            self.error()
            return None
        
        
        #Deal with Missing Data –––––––––––––––––––––––––
        print(self.color.BOLD+"\nStep 4. Dealing with Any Missing Data\n"+self.color.END)   
        
        if self.x.isnull().sum().sum() > 0:
            print('The following columns have missing data we have to deal with before modelling.')
            print('For each, you can choose how to handle the missing data.\n')
            
            i=1
            for col in self.x.columns: 
                if self.x[col].isna().sum() > 0:
                    print(str(i) +':',col,'– Missing Data Count:',self.x[col].isna().sum(),'/ '+str(len(self.x[col])))
                    #Data removal function here
                    print('For this column, would you like to...')
                    print('\t1. REMOVE the missing rows from the data set. ')
                    print('\t2. REPLACE this data with the median value for numerical data or most common for categorical data)')
                    print('\t3. KEEP the data as is (for categorical data only)')
                    choice = input(self.color.GREEN + 'Type in the assocated number to make your choice:'+ self.color.END+'\n')
                    if choice == '1':
                        print('Missing data was removed.')
                        self.x = self.x[self.x[col].notna()]
                    elif choice == '2':
                        print('Missing data was replaced.')
                        if(self.x[col].dtype == np.float64 or self.x[col].dtype == np.int64):
                            self.x[col].fillna(self.x[col].median(),inplace=True)
                        else:
                            self.x[col].fillna(self.x[col].mode(),inplace=True)
                    elif choice == '3':
                        print('Missing data was kept.')
                        self.x[col].fillna('NaN')
                    else:
                        self.error()
                        return None
                    print('')
                i+=1
                
                     
        else:
            print('No missing data found... Moving on!')
        
        
        self.y = self.x[self.y_name]
        self.x = self.x.drop(self.y_name,axis=1)
        self.scrubbed = True
        
        print('\nScrub complete! You can now access your scrubbed X and Y files at OBJECT.x and OBJECT.y respectively.')
        print('Next, call the ' + self.color.BLUE + 'explore() ' + self.color.END +'function to select your data or you can skip straight to the ' + self.color.BLUE + 'model()' + self.color.END + 'function.')
        
        
    #Explore Function
    def explore(self):   
        print('\n------------------------------------')
        print('Exploring Your Data')
        print('------------------------------------')
        print(self.color.UNDERLINE+'Please be patient... this may take a minute if you have a lot of data.'+self.color.END)

        choice = input("Would you like to do a simple explore (.head, .info, .describe) or an advanced explore (using pandas profiling)" + self.color.GREEN + ' Enter 1 for simple, or 2 for advanced.'+ self.color.END+'\n')
        if choice == '1':
            if self.scrubbed == True:
                x_eda = self.x
                x_eda['Y'] = self.y



                choice = input("\nWould you like to explore the original dataset, or the scrubbed dataset?" + self.color.GREEN + ' Enter 1 for original, or 2 for scrubbed.'+ self.color.END+'\n')
                
                if choice == '1':
                    print( '\n' + self.color.BLUE + self.color.BOLD + 'Basic Info' + self.color.END)
                    info = self.og_df.info()

                    print( '\n' + self.color.BLUE + self.color.BOLD + 'First 5 Rows' + self.color.END)
                    head = self.og_df.head()
                    display(HTML(head.to_html()))


                    print( '\n' + self.color.BLUE + self.color.BOLD + 'Descriptive Statistics' + self.color.END)
                    describe = self.og_df.describe()
                    display(HTML(describe.to_html()))                
               
                elif choice == '2':
                    print( '\n' + self.color.BLUE + self.color.BOLD + 'Basic Info' + self.color.END)
                    info = x_eda.info()

                    print( '\n' + self.color.BLUE + self.color.BOLD + 'First 5 Rows' + self.color.END)
                    head = x_eda.head()
                    display(HTML(head.to_html()))


                    print( '\n' + self.color.BLUE + self.color.BOLD + 'Descriptive Statistics' + self.color.END)
                    describe = x_eda.describe()
                    display(HTML(describe.to_html()))
                
                else:
                    self.error()
                    return None

            else:
                print('\nExploring original (unscrubbed) dataset. You can call .scrub() on the object to explore the scrubbed dataset later.')
                
                print( '\n' + self.color.BLUE + self.color.BOLD + 'Basic Info' + self.color.END)
                info = self.og_df.info()
                               
                print( '\n' + self.color.BLUE + self.color.BOLD + 'First 5 Rows' + self.color.END)
                head = self.og_df.head()
                display(HTML(head.to_html()))
                

                print( '\n' + self.color.BLUE + self.color.BOLD + 'Descriptive Statistics' + self.color.END)
                describe = self.og_df.describe()
                display(HTML(describe.to_html()))
                #print(tabulate(self.og_df, headers='keys', tablefmt='psql'))
        
            
        elif choice == '2':
          
            if self.scrubbed == True:
                x_eda = self.x
                x_eda['Y'] = self.y



                choice = input("Would you like to explore the original dataset, or the scrubbed dataset?" + self.color.GREEN + ' Enter 1 for original, or 2 for scrubbed.'+ self.color.END+'\n')
                if choice == '1':
                    profile = self.og_df.profile_report()
                elif choice == '2':
                    profile = x_eda.profile_report() 
                else:
                    self.error()
                    return None

            else:
                print('Exploring original (unscrubbed) dataset. You can call .scrub() on the object to explore the scrubbed dataset later.')


                profile = self.og_df.profile_report()

                #profile = ProfileReport(self.og_df, 
                 #           title='Pandas Profiling Report', 
                  #          html={'style':{'full_width':True}}) 
            profile.to_notebook_iframe()
        else:
            self.error()
            return None            
        
    #Model Data Function
    def model(self):
        print('\n------------------------------------')
        print('Modeling Your Data')
        print('------------------------------------')
        
        #Get Dummies
        print(self.color.BOLD + '\nGetting dummy variables...' + self.color.END)
        x_dummies = pd.get_dummies(self.x)
        print('Pre-Dummy Shape:', self.x.shape)
        print('Post-Dummy Shape:', x_dummies.shape)
                

        #Data Scaling
        print(self.color.BOLD + '\nScaling your data...' + self.color.END)
            
        print("Would you like to normalize (Min-Max scaler) or Standardize (standard scaler) the data?")
        choice = input(self.color.GREEN + 'Enter 1 for MinMax (Recommended), 2 for Standard Scaler, or 3 to skip.'+ self.color.END+'\n')
        if choice == '1':
            scaler = MinMaxScaler()
            print('MinMax Scaler Selected')
            self.scaler = MinMaxScaler()
        elif choice == '2':
            scaler = StandardScaler() 
            print('Standard Scaler Selected.')
            self.scaler = StandardScaler()
        elif choice == '3':
            print('Skipping data scaling.')
            self.scaler = None
        else:
            self.error()
            return None   
        
        if choice == '1' or choice == '2':
            x_final = self.scaler.fit_transform(x_dummies)

            print('Scaling complete.')
            print( '\nHere are the ' + self.color.BLUE + self.color.BOLD + 'first 3 rows' + self.color.END + ' to confirm the change was successfully made.')
            head = pd.DataFrame(x_final).head(3)
            display(HTML(head.to_html()))    
        else:
            x_final = x_dummies
        
        #Train Test Split
        print(self.color.BOLD + '\nSplitting up your train and test data...' + self.color.END)
        X_train, X_test, y_train, y_test = train_test_split(x_final, self.y, stratify=self.y,random_state=0)
   
        
        #Model Choices
        print(self.color.BOLD + '\nChosing Which Models to Use...' + self.color.END)
        print(self.color.RED + '\nHeads up: ' + self.color.END + 'Selecting multiple models may result in extreme wait times.')
        print('You can always start with 1-2 models and add more later.')
        print('All models previously run can be compared by calling ' + self.color.BLUE + '.interpret()'+ self.color.END + ' later.\n')
        
        model_array = [['Decision Tree',False],
                       ['Random Forrest',False],
                       ['AdaBoost',False],
                       ['XGBoost',False],
                       ['Linear Support Vector',False],
                       ['Neural Network',False]]
        
        for i in range(len(model_array)):
            print(i+1,model_array[i][0])
        
        model_choice = input(self.color.GREEN + '\nPlease type in the corrosponding number of each model you want to run, sepparating each number by a space.'+ self.color.END+'\n')
        model_choice = model_choice.split()
        model_choice = list(map(int, model_choice))
        
        for model in model_choice:
            model_array[model-1][1] = True
            print(model_array[model-1])
        
        
        print(self.color.BOLD + '\nTraining Baseline Models' + self.color.END)
        
        #Decision Tree
        if model_array[0][1] == True:
            print('\nDecision Tree Launching...')

            now = datetime.now()
            current_time = now.strftime("%D, %H:%M:%S")            

            dt_clf = DecisionTreeClassifier(random_state=0)
            dt_clf.fit(X_train, y_train)
            training_preds = dt_clf.predict(X_train)
            val_preds = dt_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
            
            self.model_comparison['Decision Tree'] = {'Model':dt_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}

        #Random Forrest
        if model_array[1][1] == True:
            print('\nRandom Forrest Launching...')
 
            now = datetime.now()
            current_time = now.strftime("%D, %H:%M:%S")            
            
            rf_clf = RandomForestClassifier(random_state=0)
            rf_clf.fit(X_train, y_train)
            training_preds = rf_clf.predict(X_train)
            val_preds = rf_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
            
            self.model_comparison['Random Forrest'] = {'Model':rf_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}
            
        #Adaboost
        if model_array[2][1] == True:
            print('\nAdaboost Launching...')
            
            now = datetime.now()
            current_time = now.strftime("%D, %H:%M:%S")            

            ab_clf = AdaBoostClassifier(random_state=0)
            ab_clf.fit(X_train, y_train)
            training_preds = ab_clf.predict(X_train)
            val_preds = ab_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)
            
            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
                        
            self.model_comparison['Adaboost'] = {'Model':ab_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}

        #XGBoost
        if model_array[3][1] == True:
            print('\nXGBoost Launching...')

            now = datetime.now()
            current_time = now.strftime("%D, %H:%M:%S")            

            xgb_clf = xgb.XGBClassifier(random_state=0)
            xgb_clf.fit(X_train, y_train)
            training_preds = xgb_clf.predict(X_train)
            val_preds = xgb_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)
          
            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
            
            self.model_comparison['XGBoost'] = {'Model':xgb_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}

        #Linear Support Vector
        if model_array[4][1] == True:
            print('\nLinear Support Vector Launching...')
            
            now = datetime.now()
            current_time = now.strftime("%D, %H:%M:%S")            
            
            svc_clf = LinearSVC(random_state=0)
            svc_clf.fit(X_train, y_train)
            training_preds = svc_clf.predict(X_train)
            val_preds = svc_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)
            
            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
                  
            self.model_comparison['Linear SVC'] = {'Model':svc_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}
        
        
        
        
        best = []
        
        for k, v in self.model_comparison.items():
            if best == []:
                best = [k, v]
            else:
                if best[1]['Test Accuracy'] < v['Test Accuracy']:
                    best = [k, v]      
        
        print("\nThe best performing model was:",self.color.PURPLE,best[0],'– with a Test Accuracy of: ',best[1]['Test Accuracy'], self.color.END)
        
        print(self.color.BOLD + '\nOptimizing Models' + self.color.END)
        
        
        #Optimization ------------------------------------------------

        
        #Grid Search for Decision Tree
        
        if model_array[0][1] == True:
            print('\nOptimizing Decision Tree...')
            param_grid = {
                "max_depth": [10,50,100,500,1000,None],
                "criterion": ["gini", "entropy"],
                "min_samples_split": [2,3,4,5],
                'random_state': [0]
            }

            grid_clf = GridSearchCV(dt_clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
            grid_clf.fit(X_train, y_train)

            best_parameters = grid_clf.best_params_

            print("Grid Search found the following optimal parameters: ")
            for param_name in sorted(best_parameters.keys()):
                print("%s: %r" % (param_name, best_parameters[param_name]))

            training_preds = grid_clf.predict(X_train)
            val_preds = grid_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

            self.model_comparison['Optimized Decision Tree'] = {'Model':grid_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}

            #print confusion matrix
            self.conf_mat('Optimized Decision Tree',val_preds,y_test)
            
        #Grid Search for Random Forrest
        if model_array[1][1] == True:
            print('\nOptimizing Random Forrest...')
            param_grid = {
                "n_estimators": [10,50,100,500],
                "max_depth": [10,50,100,500,1000,None],
                'random_state': [0]
            }

            grid_clf = GridSearchCV(rf_clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
            grid_clf.fit(X_train, y_train)

            best_parameters = grid_clf.best_params_

            print("Grid Search found the following optimal parameters: ")
            for param_name in sorted(best_parameters.keys()):
                print("%s: %r" % (param_name, best_parameters[param_name]))

            training_preds = grid_clf.predict(X_train)
            val_preds = grid_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

            self.model_comparison['Optimized Random Forrest'] = {'Model':grid_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}

            #print confusion matrix
            self.conf_mat('Optimized Random Forrest',val_preds,y_test)

        #Grid Search for AdaBoost
        if model_array[2][1] == True:
            print('\nOptimizing AdaBoost...')
            param_grid = {
                'learning_rate':[.1,.5,1],
                "n_estimators": [10,50,100,500,1000],
                'random_state': [0]
            }

            grid_clf = GridSearchCV(ab_clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
            grid_clf.fit(X_train, y_train)

            best_parameters = grid_clf.best_params_

            print("Grid Search found the following optimal parameters: ")
            for param_name in sorted(best_parameters.keys()):
                print("%s: %r" % (param_name, best_parameters[param_name]))

            training_preds = grid_clf.predict(X_train)
            val_preds = grid_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("")
            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

            self.model_comparison['Optimized AdaBoost'] = {'Model':grid_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}
            
            #print confusion matrix
            self.conf_mat('Optimized AdaBoost',val_preds,y_test)

        #Grid Search for XGBoost
        if model_array[3][1] == True:
            print('\nOptimizing XGBoost...')
            param_grid = {
                'eta':[.1,.3,.5,.9],
                'max_depth':[3,6,12],
                'lambda':[0,1],
                'alpha':[0,1],
                'random_state': [0]
            }

            grid_clf = GridSearchCV(xgb_clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
            grid_clf.fit(X_train, y_train)

            best_parameters = grid_clf.best_params_

            print("Grid Search found the following optimal parameters: ")
            for param_name in sorted(best_parameters.keys()):
                print("%s: %r" % (param_name, best_parameters[param_name]))

            training_preds = grid_clf.predict(X_train)
            val_preds = grid_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("")
            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

            self.model_comparison['Optimized XGBoost'] = {'Model':grid_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}
            
            #print confusion matrix
            self.conf_mat('Optimized XGBoost',val_preds,y_test)

        #Grid Search for Linear SVC
        if model_array[4][1] == True:
            print('\nOptimizing Linear SVC...')
            param_grid = {
                'loss':['hinge', 'squared_hinge'],
                'C':[1,10,100],
                'class_weight':[None,'balanced'],
                'max_iter':[1000,2000,3000],
                'random_state': [0]
            }

            grid_clf = GridSearchCV(svc_clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
            grid_clf.fit(X_train, y_train)

            best_parameters = grid_clf.best_params_

            print("Grid Search found the following optimal parameters: ")
            for param_name in sorted(best_parameters.keys()):
                print("%s: %r" % (param_name, best_parameters[param_name]))

            training_preds = grid_clf.predict(X_train)
            val_preds = grid_clf.predict(X_test)
            training_accuracy = accuracy_score(y_train, training_preds)
            val_accuracy = accuracy_score(y_test, val_preds)

            print("")
            print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
            print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

            self.model_comparison['Optimized Linear SVC'] = {'Model':grid_clf,'Train Accuracy':round(training_accuracy,4),'Test Accuracy':round(val_accuracy,4),'Time Run':current_time}
            
            #print confusion matrix
            self.conf_mat('Optimized Linear SVC',val_preds,y_test)            

        best = []
        
        for k, v in self.model_comparison.items():
            if best == []:
                best = [k, v]
            else:
                if best[1]['Test Accuracy'] < v['Test Accuracy']:
                    best = [k, v]     
        
        print("\nThe best performing model was:",self.color.PURPLE,best[0],'– with a Test Accuracy of: ',best[1]['Test Accuracy'], self.color.END)
        
        print('\nModeling Complete! Next, call the ' + self.color.BLUE + '.interpret()' + self.color.END + ' function.')
   
        self.best_model = best
    
    #Interpret Model Performance Function
    def interpret(self):
        print('\n------------------------------------')
        print('Interpreting Your Results')
        print('------------------------------------\n')
        
        model_name = []
        model_acc = []

        for k, v in self.model_comparison.items():
            model_name.append(str(k))
            model_acc.append(v['Test Accuracy'])  

        fig, ax  = plt.subplots(figsize = (12,6))
        
        minlim = min(list(model_acc)) - .01
        maxlim = max(list(model_acc)) + .01
        
        plt.xlabel("Models",fontsize = 15, fontweight = 'medium')
        plt.ylabel("Test Data Accuracy",fontsize = 15, fontweight = 'medium')
        
        ax.set_ylim([minlim, maxlim],auto=False)    
        ax.bar(model_name,model_acc)
        
        plt.title('Model Performance Comparison',fontsize = 20, fontweight = 'semibold')
        
        
        for i, v in enumerate(model_acc):
            ax.text(i, 
                    minlim+.003, 
                    model_acc[i], 
                    fontsize=12,
                   horizontalalignment='center',
                   color = 'white')
        
        
        plt.show()
        
        print("\nThe best performing model was:",self.color.PURPLE,self.best_model[0],'– with a Test Accuracy of: ',self.best_model[1]['Test Accuracy'], self.color.END)
        
    def predict(self):
        print('\n------------------------------------')
        print('Making a Prediction')
        print('------------------------------------')
        
        print(self.color.BOLD+'\nPlease provide the input values to make a prediction. (MUST be correct data type)'+self.color.END)

        inputs = []
        for col in self.x.columns:
            val = input('Value for: '+ self.color.BLUE + str(col) + self.color.END +'?' + ' (Type = ' + str(self.x[col].dtypes) + ')' + ' (Example = ' + str(self.x[col].iloc[0]) + ')' )
            
            if(self.x[col].dtype == np.float64):
                val = float(val)
            if(self.x[col].dtype == np.int64):
                val = int(val)
            inputs.append(val)

        
        inputs = [inputs]
        
        #getting dummies
        input_dummies = pd.get_dummies(pd.DataFrame(inputs))
        
        #scaling inputs
        if self.scaler != None:
            final_inputs = self.scaler.transform(input_dummies)
        
        #Back to numpy
        #final_inputs = final_inputs.to_numpy()
        
        print(self.color.BOLD+'\nBest Performing Model Prediction'+self.color.END)
        print(self.best_model[0]+ ' predicts --> ' + self.color.BLUE + str(self.best_model[1]['Model'].predict(final_inputs)[0]) + self.color.END)
        
        print(self.color.BOLD+'\nAll Model Predictions'+self.color.END)
        all_preds = []
        for k, v in self.model_comparison.items():
            print(str(k)+ ' predicts -->', self.color.BLUE + str(v['Model'].predict(final_inputs)[0]) + self.color.END)    
            all_preds.append(v['Model'].predict(inputs)[0])
        
        try:
            print(self.color.BOLD+'\nMost Common Prediction '+self.color.END + '--> '+ self.color.BLUE+ str(mode(all_preds)) + self.color.END)
        except StatisticsError:
            print (self.color.BOLD+'\nNo unique mode found.'+self.color.END)