from bson.py3compat import reraise_instance
from pymongo import MongoClient , ASCENDING, collation
import geopandas as gpd
import os , shutil
from datetime import date, datetime
import random

DATABASE_NAME = "DataBase"


# shapes original --> shapeFile_original
# classified shapes --> shapeFile_classified
# classified tiles -> out_tiles



def dropCollection(db,name):


    collection = db[name]
    for i in collection.find():
        path = i['path']
        path = path + "\.."
        folderPath = os.path.normpath(path)
        if(os.path.exists(folderPath)):
            for filename in os.listdir(folderPath):
                file_path = os.path.join(folderPath, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
      
            os.removedirs(folderPath)
    collection.drop()



def createShapeFiles(db):
    fileName = "EGY_adm1.shp"
    fullpath = os.path.join(os.getcwd() , 'Data' , 'shapefile_city' , 'Original_shapes')
    gdf = gpd.read_file(os.path.join(fullpath, fileName))
    boundries = gdf.bounds
    gdf = gdf.to_crs(32636)
    compare_bounds = gdf.bounds
    gdf = gdf.to_crs("EPSG:4326")
    Date = datetime.date(datetime.now())
    for index, row in gdf.iterrows():
        governetName = row["NAME_1"]
        governetName = governetName.replace('/' ,' ')
        governetBounds = list(boundries.iloc[index,:])
        governetBoundsCopared = list(compare_bounds.iloc[index,:])
        governateShape = row["geometry"]
        data = {'City' : [governetName] , 'geometry' : [governateShape]}
        newGDF = gpd.GeoDataFrame(data)
        os.mkdir(os.path.join(fullpath , governetName))
        newGDF.to_file(os.path.join(fullpath , governetName , governetName+".shp"), driver='ESRI Shapefile')
        objectData = gpd.GeoSeries([governateShape]).__geo_interface__
        shapeType = objectData["features"][0]['geometry']['type']
        shapeCord = objectData["features"][0]['geometry']['coordinates']
        db.shapeFile_original.insert_one({
                        "name" : governetName.replace(" ","-"),
                        "path" : os.path.join(fullpath , governetName , governetName+".shp"),
                        "boundries" :governetBounds,
                        "Date" :   str(datetime.date(datetime.now())),
                        "Type" : shapeType,
                        "cord" : shapeCord,
                        "compare_bounds" : governetBoundsCopared
                    })

def createResults(db):
    for i in db.shapeFile_original.find():
        db.shapeFile_classified.insert_one(
            {
        "name": i["name"],
        "path": i["path"],
        "Original_shape_id": i["_id"],
        "boundries": i["boundries"],
        "resource": {
            "urban-land": 4863 * random.randint(0,20),
            "agriculture_land": 4896 * random.randint(0,20),
            "aqua": 7893 * random.randint(0,20),
            "trees": 4896 * random.randint(0,20),
            "water": 13654 * random.randint(0,20),
            "sand-rocks": 8200 * random.randint(0,20),
            "unknown": 2 * random.randint(0,5),
            "road": 4896 * random.randint(0,20)
        },
        "Date": i["Date"],
        "Type" : i["Type"],
        "cord" : i["cord"]

        }
    )

def main():
    try:
        
        client = MongoClient('mongodb://localhost:27017/')
        db = client[DATABASE_NAME]

        dropCollection(db , 'shapeFile_original')
        dropCollection(db , 'shapeFile_classified')

        createShapeFiles(db)
        createResults(db)

        # dropCollection(db , 'out_tiles')

        print("_______database has been reset _______")
    except():
        print("there was an error")


    

if __name__ == '__main__':
    main()