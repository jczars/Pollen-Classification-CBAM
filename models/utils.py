# -*- coding: utf-8 -*-

import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dataSet(_path_data, _csv_data, CATEGORIES):
  """
  --> create data set in format csv
  :param: _path_data: path the dataSet
  :param: _csv_data: path with file name '.csv'
  :param: _categories: the classes the data
  :return: DataFrame with file path
  """
  data=pd.DataFrame(columns= ['file', 'labels'])
  print('_path_data ', _path_data)
  print('_csv_data ', _csv_data)
  print('CATEGORIES: ', CATEGORIES)

  c=0
  #cat_names = os.listdir(_path_data)
  for j in CATEGORIES:
      pathfile = _path_data+'/'+j
      filenames = os.listdir(pathfile)
      for i in filenames:
        #print(_path_data+'/'+j+'/'+i)
        data.loc[c] = [str(_path_data+'/'+j+'/'+i), j]
        c=c+1
  #print(c)
  data.to_csv(_csv_data, index = False, header=True)
  data_csv = pd.read_csv(_csv_data)
  print(_csv_data)
  print(data_csv.groupby('labels').count())

  return data

def create_unlabelSet(_unlabels):
  unlabelsBD=pd.DataFrame(columns= ['file'])
  c=0
  filenames = os.listdir(_unlabels)
  for i in filenames:
    #print(_unlabels+'/'+i)
    unlabelsBD.loc[c] = [str(_unlabels+'/'+i)]
    c=c+1
  print(c)
  return unlabelsBD

def create_folders(_save_dir, flag=1):
  """
  -->create folders
  :param: _save_dir: path the folder
  :param: flag: rewrite the folder, (1 for not and display error: 'the folder already exists)
  """
  if os.path.isdir(_save_dir):
      if flag:
          raise FileNotFoundError("folders test already exists: ", _save_dir)
      else:
          print('folders test already exists: ', _save_dir)
  else:
      os.mkdir(_save_dir)
      print('create folders test: ', _save_dir)


def criarTestes(_path, nmTeste):
  """
  -->create folders
  :param: _path: path the folder
  :param: nmTeste: name the test
  """
  # nome do teste: modelo+size+aug
  _dir_test = _path+'/Train_'+nmTeste+'/'
  print('folders test: ',_dir_test)
  create_folders(_dir_test, flag=0)
  return _dir_test, nmTeste

def renomear_path(conf, data_uns, verbose=0):
  unlabels_path = f"{conf['path_base']}/images_unlabels/"

  print(data_uns.head())

  for index, row in data_uns.iterrows():
    values=row['file']
    if verbose>0:
        print('[INFO] renomear_path')
        print(values)
    isp=values.split('/')
    new_path=unlabels_path+'unlabels/'+isp[-1]
    data_uns.at[index,'file']=new_path
    if verbose>0:
        print(new_path)
      

def get_process_memory():
    """Return total memory used by current PID, **including** memory in shared libraries
    """
    raw = os.popen(f"pmap {os.getpid()}").read()
    # The last line of pmap output gives the total memory like
    # " total            40140K"
    memory_mb = int(raw.split("\n")[-2].split()[-1].strip("K")) // 1024
    return memory_mb

def bytes_to_mb(size_in_bytes):
    size_in_mb = size_in_bytes / (1024.0 ** 2)
    return size_in_mb

def select_pseudos(pseudos, CATEGORIES, menor, _tempo):
    df_sel=pd.DataFrame(columns= ['file','labels'])

    df_cat_size=[]
    print(f"cat{'-':<17}| total")
    print("-"*30)
    for cat in CATEGORIES:
        df = pseudos[pseudos['labels'] == cat]
        if len(df)>0:
            df=df[:menor]
            size=len(df)
            print(f"{cat:<20}| {size}")
            df_cat_size.append([_tempo, cat, size])
            df_sel=pd.concat([df_sel,df])
            _size_selec=len(df_sel)
            print('Total de dados selecioandos ', _size_selec)
    return df_sel, df_cat_size
def add_row_csv(filename_csv, data):
    """
    Add a new row to a CSV file.
    
    Args:
        filename_csv (str): Name of the CSV file.
        data (list): Data to be inserted into the CSV file.
    """
    with open(filename_csv, 'a') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerows(data)


def graph_img_cat0(data_dir):
    """
    Generates a bar chart showing the number of images per category.

    Args:
        data_dir (str): Directory where the category subfolders are located.

    Returns:
        matplotlib.figure.Figure: A figure object showing the number of images per category.
    """
    # Ensure the folders within the directory are indeed subfolders
    category_names = [category for category in sorted(os.listdir(data_dir))
                      if os.path.isdir(os.path.join(data_dir, category))]

    img_pr_cat = []

    # Count the number of images per category
    for category in category_names:
        category_path = os.path.join(data_dir, category)
        img_pr_cat.append(len([f for f in os.listdir(category_path) if f.endswith(('jpg', 'png', 'jpeg'))]))

    # Create the bar chart
    fig = plt.figure(figsize=(15, 10))
    sns.barplot(y=category_names, x=img_pr_cat).set_title("Number of training images per category:")

    return fig
import os
import matplotlib.pyplot as plt
import seaborn as sns

import os
import matplotlib.pyplot as plt
import seaborn as sns

def graph_img_cat(data_dir):
    """
    Generates a bar chart showing the number of images per category.

    Args:
        data_dir (str): Directory where the category subfolders are located.

    Returns:
        matplotlib.figure.Figure: A figure object showing the number of images per category.
    """
    # List categories (subfolders)
    category_names = [category for category in sorted(os.listdir(data_dir))
                      if os.path.isdir(os.path.join(data_dir, category))]

    img_pr_cat = []

    # Count the number of images per category
    for category in category_names:
        category_path = os.path.join(data_dir, category)
        img_pr_cat.append(len([f for f in os.listdir(category_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]))

    # Create the horizontal bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    bars = sns.barplot(y=category_names, x=img_pr_cat, ax=ax)

    # Add values on top of each bar
    for i, bar in enumerate(bars.patches):
        ax.text(
            bar.get_width() + 1,  # Horizontal position of the text
            bar.get_y() + bar.get_height() / 2,  # Centered vertically on the bar
            f"{img_pr_cat[i]}",  # Text value
            va="center", ha="left", fontsize=10, color="black"
        )

    # Adjust titles and labels
    ax.set_title("Number of training images per category", fontsize=16)
    ax.set_xlabel("Number of images", fontsize=14)
    ax.set_ylabel("Categories", fontsize=14)
    ax.set_xlim(0, max(img_pr_cat) * 1.1)  # Add margin to the X axis

    # Adjust layout
    plt.tight_layout()

    return fig


   
if __name__=="__main__":
   help(create_dataSet)
   help(create_unlabelSet)
   help(create_folders)
   help(criarTestes)
   help(add_row_csv)
   