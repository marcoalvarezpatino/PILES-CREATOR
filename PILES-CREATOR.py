#adding new field library needed
from PyQt5.QtCore import QVariant
from qgis.core import *
import pandas as pd
import numpy as np

# seting inputs

# number of piles variable
pile_total_input = 7

pile_count = pile_total_input - 1

# setting minimum and maximum pile reveal variable
min_reveal = 4

max_reveal = 6

# creating a polygon from a QGIS layer input called Tracker_Polylines
input_polylines = QgsProject.instance().mapLayersByName("Tracker_Polylines")[0]

# create a polygon / polyline check here / error message

# get crs for vector layer
crs = input_polylines.crs().authid()

# creating polygons from input polylines
polygons = processing.run("qgis:linestopolygons", {'INPUT': input_polylines,
                                                   'OUTPUT': 'TEMPORARY_OUTPUT'})
polygons = polygons['OUTPUT']

# creating a  reference variable
layer = polygons

# lets delete all attributes and add in fields
attribute_list = layer.attributeList()
layer_provider = layer.dataProvider()
layer.startEditing()
layer_provider.deleteAttributes(attribute_list)
layer.commitChanges()

# add a field called Tracker_ID Integer class

# empty creates a variable to be updated
layer_provider.addAttributes([QgsField("Tracker_ID", QVariant.Int)])
layer.updateFields()

# UPDATING/ADD ATTRIBUTE VALUE the 4 represents the field column of the layer
features = layer.getFeatures()
layer.startEditing()
for f in features:
    id = f.id()
    attr_value = {0: id}
    layer_provider.changeAttributeValues({id: attr_value})
layer.commitChanges()

# loading in the resulting Tracker polygons as a layer
# load layer
QgsProject.instance().addMapLayer(layer).setName('Tracker_ID')

# extract vertices
vertices = processing.run("native:extractvertices", {'INPUT': layer, 'OUTPUT': 'TEMPORARY_OUTPUT'})

v1 = vertices['OUTPUT']

# adding geometry values to the vertices
v1 = processing.run("qgis:exportaddgeometrycolumns", {'INPUT': v1, 'CALC_METHOD': 0, 'OUTPUT': 'TEMPORARY_OUTPUT'})
v1 = v1['OUTPUT']
QgsProject.instance().addMapLayer(v1).setName('v1')

# listing all the columns to include inside the data frame

cols = ['Tracker_ID', 'xcoord', 'ycoord']

# A generator yielding one line at a time
datagen = ([f[col] for col in cols] for f in v1.getFeatures())

# dataframe from records
df = pd.DataFrame.from_records(data=datagen, columns=cols)

# get max x and max y values for each Tracker
df['max_x'] = df.groupby('Tracker_ID')['xcoord'].transform('max')
df['max_y'] = df.groupby('Tracker_ID')['ycoord'].transform('max')
df['min_x'] = df.groupby('Tracker_ID')['xcoord'].transform('min')
df['min_y'] = df.groupby('Tracker_ID')['ycoord'].transform('min')
df['center_x'] = (df['max_x'] + df['min_x']) / 2
df['distance'] = (df['max_y'] - df['min_y']) / pile_count

# df2['Y'] = df['max_y'] *

# get list of trackers to run through a loop
tracker_list = df['Tracker_ID'].unique().tolist()

index_list = np.arange(0, pile_total_input)

final_df = pd.DataFrame(columns=['pile_id', 'Tracker_ID', 'x', 'y'])

for tracker in tracker_list:
    center_x = df[df['Tracker_ID'] == tracker]['center_x'].iloc[0]
    north_y = df[df['Tracker_ID'] == tracker]['max_y'].iloc[0]
    distance = df[df['Tracker_ID'] == tracker]['distance'].iloc[0]
    df_temp = pd.DataFrame({'pile_id': index_list, 'Tracker_ID': tracker, 'x': center_x})
    df_temp['y'] = north_y - df_temp['pile_id'] * distance
    final_df = pd.concat([final_df, df_temp], ignore_index=True)

print(final_df.head())

# SAVE TO NEW CSV
# final_df.to_csv('finalnew.csv')

# Define the layer type (in this case, a Point layer)

layer = QgsVectorLayer('Point?crs= + crs + ', 'MyLayer', 'memory')

# Add fields (columns) to the layer
provider = layer.dataProvider()
provider.addAttributes([
    QgsField('Tracker_ID', QVariant.Int),
    QgsField('x', QVariant.Double),
    QgsField('y', QVariant.Double)
])
layer.updateFields()

# Add features to the layer
for index, row in final_df.iterrows():
    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(row['x'], row['y'])))
    feature.setAttributes([row['Tracker_ID'], row['x'], row['y']])
    provider.addFeature(feature)

# creating an variable for an input raster using a layer called EG_input, bring in you EG DTM as a tiff ard rename it as topo_input_raster in qgis
input_raster = QgsProject.instance().mapLayersByName("EG")[0]

# sampling raster values (elevations) at the location of piles base
sampled = processing.run("native:rastersampling",
                         {'INPUT': layer,
                          'RASTERCOPY': input_raster,
                          'COLUMN_PREFIX': 'z terrain enter',
                          'OUTPUT': 'TEMPORARY_OUTPUT'})

# VARIABLE CREATED USING OUTPUT OF THE RASTERSAMPLING PROCESS
layer = sampled['OUTPUT']

# renaming a field
for field in layer.fields():
    if field.name() == 'z terrain enter1':
        with edit(layer):
            idx = layer.fields().indexFromName(field.name())
            layer.renameAttribute(idx, 'z terrain enter')

QgsProject.instance().addMapLayer(layer)

# List all columns you want to include in the dataframe. I include all with:
cols = [f.name() for f in layer.fields()]

# A generator to yield one row at a time
datagen = ([f[col] for col in cols] for f in layer.getFeatures())

df = pd.DataFrame.from_records(data=datagen, columns=cols)

df['min_reveal'] = min_reveal
df['max_reveal'] = max_reveal

def linear_regress(df, verbose=False):
    """
    function to apply linear regression to each tracker in a data frame

    :param Pandas DataFrame df: df outputed from QGIS
    :param verbose: tells the program to print information about each regression

    :return Pandas DataFrame df: df of inputted df with added columns for regression parameters

    """

    # get list of trackers to run through a loop
    tracker_list = df['Tracker_ID'].unique().tolist()
    for i in range(len(tracker_list)):
        tracker = tracker_list[i]

        if verbose:
            print(f'Regressing on tracker number {tracker}')

        # x values are the northing (y) and regress on the z_terrain_enter
        x = df.loc[df['Tracker_ID'] == tracker, 'y'].to_numpy()
        y = df.loc[df['Tracker_ID'] == tracker, 'z terrain enter'].to_numpy()
        coeffic = np.polyfit(x, y, 1,)
        # plot_outcome(x, y, coeffic)

        # update df
        df.loc[df['Tracker_ID'] == tracker, 'slope'] = coeffic[0]
        df.loc[df['Tracker_ID'] == tracker, 'intercept'] = coeffic[1]

    # update y_regression now that all slopes & intercepts have been determined
    df['y_regression'] = df['slope'] * df['y'] + df['intercept']

    return df

def calculate_cf(df_trackers):
    """
    function to calculate cut and fill per pile for entered df
    :param df_trackers: df of 1+ trackers to fill in columns for
    :return: df with freshly calculated columns
    """

    # calculate tabletop elev based on a median offset
    df_trackers['offset'] = (df_trackers['max_reveal'] + df_trackers['min_reveal']) / 2
    df_trackers['Tabletop_Elev'] = df_trackers['y_regression'] + df_trackers['offset']

    # Calculate Z terrain min and max elev based on tabletop elev & min/max reveal
    df_trackers['z terrain min elev'] = df_trackers['Tabletop_Elev'] - df_trackers['max_reveal']
    df_trackers['z terrain max elev'] = df_trackers['Tabletop_Elev'] - df_trackers['min_reveal']

    # Calculate CF based on z terrain elev vs z terrain min & max
    # place 0 everywhere and then only update 0 where the following conditions are met
    df_trackers['cf'] = 0
    df_trackers.loc[df_trackers['z terrain enter'] > df_trackers['z terrain max elev'], 'cf'] = \
        df_trackers['z terrain max elev'] - df_trackers['z terrain enter']
    df_trackers.loc[df_trackers['z terrain enter'] < df_trackers['z terrain min elev'], 'cf'] = \
        df_trackers['z terrain min elev'] - df_trackers['z terrain enter']

    # setting pg based on cf
    df_trackers['pg'] = df_trackers['z terrain enter'] + df_trackers['cf']

    # converting strings to decimals
    df_trackers['Tabletop_Elev'] = pd.to_numeric(df_trackers['Tabletop_Elev'], errors='coerce')
    df_trackers['pg'] = pd.to_numeric(df_trackers['pg'], errors='coerce')
    # calculating pile reveal and rounding to 2 decimal places
    df_trackers['Pile_reveal'] = df_trackers['Tabletop_Elev'] - df_trackers['pg']
    df_trackers['Pile_reveal'] = df_trackers.Pile_reveal.round(2)
    df_trackers['slope_label'] = df_trackers['slope'] * 100
    df_trackers['slope_label'] = df_trackers.slope_label.round(2)

    # get count of piles where CF is necessary
    count_CF_piles_calc = len(df_trackers[df_trackers['cf'] != 0])

    # get total CF
    total_cf_cal = sum(abs(df_trackers['cf']))

    return df_trackers

df_trackers = linear_regress(df, verbose=False)
df = calculate_cf(df_trackers)
print(df.head())

print('Linear Regression Complete')



