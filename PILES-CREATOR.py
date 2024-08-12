#adding new field library needed
from PyQt5.QtCore import QVariant
from qgis.core import *
import pandas as pd
import numpy as np

#creating a polygon from a QGIS layer input called Tracker_Polylines
input_polylines = QgsProject.instance().mapLayersByName("Tracker_Polylines")[0]

#create a polygon / polyline check here / error message

#creating polygons from input polylines
polygons = processing.run("qgis:linestopolygons", {'INPUT':input_polylines,
                                                   'OUTPUT':'TEMPORARY_OUTPUT'})
polygons = polygons['OUTPUT']

# number of piles variable
pile_total_input = 7

pile_count = piles_total_input - 1
    

#creating a  reference variable
layer = polygons

# lets delete all attributes and add in fields
attribute_list = layer.attributeList()
layer_provider = layer.dataProvider()
layer.startEditing()
layer_provider.deleteAttributes(attribute_list)
layer.commitChanges()

#add a field called Tracker_ID Integer class

#empty creates a variable to be updated 
layer_provider.addAttributes([QgsField("Tracker_ID",QVariant.Int)])
layer.updateFields()

#UPDATING/ADD ATTRIBUTE VALUE the 4 represents the field column of the layer
features = layer.getFeatures()
layer.startEditing()
for f in features:
    id=f.id()
    attr_value={0:id}
    layer_provider.changeAttributeValues({id:attr_value})
layer.commitChanges()

#loading in the resulting Tracker polygons as a layer
#load layer
QgsProject.instance().addMapLayer(layer).setName('Tracker_ID')

#extract vertices
vertices = processing.run("native:extractvertices", {'INPUT':layer,'OUTPUT':'TEMPORARY_OUTPUT'})

v1 = vertices['OUTPUT']

#adding geometry values to the vertices

v1 = processing.run("qgis:exportaddgeometrycolumns", {'INPUT':v1,'CALC_METHOD':0,'OUTPUT':'TEMPORARY_OUTPUT'})
v1 = v1['OUTPUT']
QgsProject.instance().addMapLayer(v1).setName('v1')

#listing all the columns to include inside the data frame

cols = ['Tracker_ID', 'xcoord', 'ycoord']

#A generator yielding one line at a time
datagen = ([f[col] for col in cols] for f in v1.getFeatures())

#dataframe from records
df = pd.DataFrame.from_records(data=datagen, columns=cols)

#get max x and max y values for each Tracker
df['max_x'] = df.groupby('Tracker_ID')['xcoord'].transform('max')
df['max_y'] = df.groupby('Tracker_ID')['ycoord'].transform('max')
df['min_x'] = df.groupby('Tracker_ID')['xcoord'].transform('min')
df['min_y'] = df.groupby('Tracker_ID')['ycoord'].transform('min')
df['center_x'] = (df['max_x']+ df['min_x'])/2
df['distance'] = (df['max_y']-df['min_y'])/ pile_count


# df2['Y'] = df['max_y'] *

# get list of trackers to run through a loop
tracker_list = df['Tracker_ID'].unique().tolist()

index_list = np.arange(0, pile_total_input)

final_df = pd.DataFrame(columns = ['pile_id','Tracker_ID','x','y'])

for tracker in tracker_list:
    center_x = df[df['Tracker_ID'] == tracker][ 'center_x'].iloc[0]
    north_y = df[df['Tracker_ID'] == tracker][ 'max_y'].iloc[0]
    distance = df[df['Tracker_ID'] == tracker][ 'distance'].iloc[0]
    df_temp = pd.DataFrame({'pile_id': index_list, 'Tracker_ID': tracker, 'x': center_x })
    df_temp['y'] = north_y - df_temp['pile_id']*distance
    final_df = pd.concat([final_df,df_temp],ignore_index= True)




print(final_df.head())
print()

#SAVE TO NEW CSV
# final_df.to_csv('finalnew.csv')

#next steps: create points file out of the data frame for each pile

# Creation of my QgsVectorLayer with no geometry
temp = QgsVectorLayer("none","result","memory")
temp_data = temp.dataProvider()
# Start of the edition
temp.startEditing()

# Creation of my fields
for head in final_df :
    myField = QgsField( head, QVariant.Double )
    temp.addAttribute(myField)
# Update
temp.updateFields()

# Addition of features
# [1] because i don't want the indexes
for row in final_df.itertuples():
    f = QgsFeature()
    f.setAttributes([row[1],row[2]])
    temp.addFeature(f)
    print(row)
# saving changes and adding the layer
temp.commitChanges()
QgsProject.instance().addMapLayer(temp).setName('temp')








# #create a new point that exists top middle and bottom middle per each tracker_id 2 points top and bottom
# #may need a for / while loop here
#
#
# expression1 = QgsExpression('maximum("ycoord", "Tracker_ID")')
# expression2 = QgsExpression('maximum("xcoord", "Tracker_ID")')
# expression3 = QgsExpression('minimum("xcoord", "Tracker_ID")')
# expression4 = QgsExpression('minimum("ycoord", "Tracker_ID")')
#
#
# #could use pandas group by but that would need me to turn the layer into a dataframe. could do that
#
# #ADDING EMPTY VARIABLES FOR FIELDS
# pr = v1.dataProvider()
# pr.addAttributes([QgsField("ymax", QVariant.Double),
#                 QgsField("xmax", QVariant.Double),
#                 QgsField("ymin", QVariant.Double),
#                 QgsField("x_med", QVariant.Double),
#                 QgsField("xmin", QVariant.Double)])
# v1.updateFields()
# #to execute expressions you need an appropriate QgsExpressions context
# context = QgsExpressionContext()
# context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(v1))
#
# with edit(v1):
#     for f in v1.getFeatures():
#         context.setFeature(f)
#         f['ymax'] = expression1.evaluate(context)
#         f['xmax'] = expression2.evaluate(context)
#         f['xmin'] = expression3.evaluate(context)
#         f['ymin'] = expression4.evaluate(context)
#         f['x_med'] = ((f['xmax']-f['xmin'])/2) + f['xmin']
#         v1.updateFeature(f)
#
# #end by creating a line vector layer at the center of the vertices
#
# QgsProject.instance().addMapLayer(v1).setName('v1')

#creating a new line named temp from our v1 point vector
#
# v2 = QgsVectorLayer("Point","temp","memory")
# pr = v2.dataProvider()
# pr.addAttributes([QgsField("Tracker_ID", QVariant.Int)])
# v2.updateFields()
#might need a dictionary or something here to try to
# with edit(v2):
#     for f in v2.getFeatures():

