from threading import Condition
from warnings import catch_warnings
from flask import Flask , jsonify , request
import flask
from flask_pymongo import PyMongo
import json
import os
from datetime import date, datetime
import shutil
import pymongo
import rasterio
import geopandas as gpd
from geotools.geomask import GeoMask 
import re
from flask_cors import CORS
import random
from osgeo import gdal
from predict.run import predict_tile, predict_and_call_backend
import threading

# Current date time in local system

# will be invoked from the enpoint where the operator send the confirmation of the file being in the Classified_tiles
# out_tiles = {
#     id : can be from us or generated from the pyMnogo ,
#     path : sting,
#     boundries : array
#     Data : Date
# }

# will be seeded in the database each shape file will be in the shapefile_city original ones (will be filled using script)
# shapeFile_original = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     path : string,
#     boundries : array,
#     Date : Date
# } 
    # 1 x M

# will be invoked after the part of the code that returnes each percentage of each shapeFile
# shapeFile_classified = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     path : string,
#     Original_shape_id : id of shapeFile_original,
#     boudnies L array,
#     resource : {dictionary of resoruces , key is the resource name and the value is the m^2}
#     date : Date
# }

# will be seeded in the database each shape file will be in the shapefile_in_city original ones (will be filled using script)
# shapeFile_Child_original = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     Original_shape_id : id of shapeFile_original,
#     path : string,
#     boundries : array,
#     Date : Date
# }

# shapeFile_Child_classified = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     shapeFile_classified_id : id of shapeFile_classified,
#     path : string,
#     boundries : array,
#     Date : Date
#     resource : {dictionary of resoruces , key is the resource name and the value is the m^2}
# }

def checkBound(shapeFile_bounds , tile_bounds):
    return  shapeFile_bounds[0] > tile_bounds[0] and \
            shapeFile_bounds[1] > tile_bounds[1] and \
            shapeFile_bounds[2] < tile_bounds[2] and \
            shapeFile_bounds[3] < tile_bounds[3] 

app = Flask(__name__ , static_folder='../build', static_url_path='/')
app.config["MONGO_URI"] = "mongodb://localhost:27017/DataBase"
mongo = PyMongo(app)
CORS(app)

def run_prediction(path):
    print(path)

@app.route('/')
def index():
    return app.send_static_file('index.html')
    
@app.route("/seedItem" , methods = ["POST"])
def seedOneIteam():
    filename = "mash_abohomos.shp"
    filepath = "I:\programming 2\AIC project\V4\web_files"
    fullpath = os.path.join(filepath, filename)
    gdf = gpd.read_file(fullpath)   
    bboundries = gdf.bounds
    b = bboundries.values[0]
    b = list(b)
    gdf = gdf.to_crs(32636)
    compare_bounds = gdf.bounds
    a = compare_bounds.values[0]
    a = list(a)
    gdf = gdf.to_crs("EPSG:4326")
    objectData = gpd.GeoSeries([gdf['geometry'][0]]).__geo_interface__
    shapeType = objectData["features"][0]['geometry']['type']
    shapeCord = objectData["features"][0]['geometry']['coordinates']
    mongo.db.shapeFile_original.insert_one({
                        "name" : filename.split(".")[0],
                        "path" : fullpath,
                        "boundries" :b,
                        "Date" :   str(datetime.date(datetime.now())),
                        "Type" : shapeType,
                        "cord" : shapeCord,
                        "compare_bounds" : a
                    })
    return "ok" , 200

@app.route("/seedShapes" , methods = ["GET"])
def seedShapes():
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
        mongo.db.shapeFile_original.insert_one({
                        "name" : governetName.replace(" ","-"),
                        "path" : os.path.join(fullpath , governetName , governetName+".shp"),
                        "boundries" :governetBounds,
                        "Date" :   str(datetime.date(datetime.now())),
                        "Type" : shapeType,
                        "cord" : shapeCord,
                        "compare_bounds" : governetBoundsCopared
                    })
    return "ok" , 200

@app.route("/makeResult" , methods = ["GET"])
def makeResult():
    for i in mongo.db.shapeFile_original.find():
        mongo.db.shapeFile_classified.insert_one(
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
      
    return "ok", 200

@app.route("/savedata" , methods = ["POST"])
def saveData():
    # we can make a Get request to view a form page on request.
    if(request.method == "POST"):
        print("here-1")
        print("Calling post and proceeding.....................")
        classified_path = os.path.join("Data","Classified_tiles")
        fileName = request.form.get("file_name")
        print('-----------------',fileName)
        print('-----------------',classified_path)

        fileOperator_path = request.form.get("file_path")
        fullpath = os.path.join(os.getcwd() , classified_path)
        print('-----------------', fileOperator_path)
        print('here 0')
        FileIsThere = os.path.isfile(fileOperator_path)
        print("here0.1")
        if(FileIsThere):
            print("here 1")
            match = re.search(r'\d{4}-\d{2}-\d{2}', fileName)
            Date = datetime.strptime(match.group(), '%Y-%m-%d').date()
            folderIsThere = os.path.isdir(os.path.join(fullpath , str(Date)))
            if(not(folderIsThere)):
                os.mkdir(os.path.join(fullpath , str(Date)))
            newPath = shutil.copy(fileOperator_path, os.path.join(fullpath , str(Date) , fileName))
            # os.remove(fileOperator_path)
            rasterBand = rasterio.open(os.path.join(fullpath , str(Date) , fileName))
            bound_data = list(rasterBand.bounds)
            retunred_object = mongo.db.out_tiles.find_one({"name" : fileName , "date" : str(Date)})
            print("here2")
            if(retunred_object == None):
                print("here3")
                db_response = mongo.db.out_tiles.insert_one({
                    "path" : str(os.path.join(fullpath , str(Date) , fileName)),
                    "name" : fileName,
                    "bounds" : bound_data,
                    "date" : str(Date)
                })
                print("here4")
                for i in mongo.db.shapeFile_original.find():
                    condition = checkBound(i["compare_bounds"] , bound_data)
                    if(condition):
                        tile_path = os.path.join(fullpath , str(Date) , fileName)
                        obj = GeoMask(tile_path,i["path"])
                        stats , path , name = obj.mask()
                        mongo.db.shapeFile_classified.insert_one({
                            "name" : i['name'],
                            "path" : path,
                            "Original_shape_id" : i["_id"],
                            "boundries" :i["boundries"],
                            "resource" : stats,
                            "Date" :   str(Date),
                            "imgURI" : 'static/imgs/' + i['name']+ ".png"
                        })

                        options_list = [
                            '-ot UInt32',
                            '-of PNG',
                            '-b 1',
                            '-expand RGBA',
                            '-scale',
                            '-outsize 1920 1080'
                        ]           

                        options_string = " ".join(options_list)
                            
                        gdal.Translate(
                            '..\\build\\static\\imgs\\' + i['name']+ ".png",
                            path,
                            options=options_string
                        )
                print("here5")

            if(newPath):
                return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
            else:
                return 'internal server error', 500
        else:
            print("didn't even enter post")
            return 'bad request!', 400

@app.route("/resources/<name>" , methods = ["GET"])
def GetResorces(name):
    data = None
    for i in mongo.db.shapeFile_classified.find({"name" : name}).sort([("Date" , pymongo.DESCENDING)]):
        data = i
        break
    if(data == None):
         return 'not Found !', 404
    else:
        data.pop("_id",None)
        data.pop("Original_shape_id",None)
        return flask.jsonify(data)


@app.route("/resources/<name>/<date>" , methods = ["GET"])
def GetResorces_by_date(name,date):
    date_string = str(date)
    format = "%Y-%m-%d"
    try:
        datetime.strptime(date_string, format)
    except:
        return 'bad request!', 400
    
    data = mongo.db.shapeFile_classified.find_one({"name" :name , "Date" : str(date)})
    if(data == None):
        return flask.jsonify({})
    
    data.pop("_id",None)
    data.pop("Original_shape_id",None)
    data.pop("path",None)
    
    return flask.jsonify(data)


@app.route("/resources/all" , methods = ["GET"])
def Get_all_Resorces():
    data = {}
    list_of_resources = []
    for i in mongo.db.shapeFile_classified.find().sort([("Date" , pymongo.DESCENDING)]):
        list_of_resources.append(i["resource"])
    for i in list_of_resources:
        for j in i:
            if(j in data):
                data[j] = data[j] + i[j]   

            else:
                data[j] = 0 
                data[j] = data[j] + i[j]
    return flask.jsonify(data)



@app.route("/predict" , methods=["POST"])
def predict():
    config = request.form.get("config")
    checkpoint_file = request.form.get("checkpoint_file")
    bands_dir = request.form.get("bands_dir")
    shape_file_path = request.form.get("shape_file_path")
    results_path = request.form.get("results_path")
    temp_directory = request.form.get("temp_directory")

    print("config",config)
    print("checkpoint_file",checkpoint_file)
    print("bands_dir",bands_dir)
    print("shape_file_path",shape_file_path)
    print("results_path",results_path)
    print("temp_directory",temp_directory)


    cmdString = "python run.py" + " --config " + config + " --checkpoint_file " + checkpoint_file + " --bands_dir " + bands_dir + " --shape_file_path " + shape_file_path + " --results_path " + results_path + " --temp_directory " + temp_directory
    print(cmdString)
    #os.chdir("/home/developer-5/Desktop/agri-react/api/predict")    
    # os.system(cmdString)
    #predict_tile(config,checkpoint_file,bands_dir,shape_file_path,temp_directory,results_path)
    #save file API
    # predict and save file
    t1 = threading.Thread(target = lambda:\
        predict_and_call_backend(config,checkpoint_file,\
                                bands_dir,shape_file_path,\
                                temp_directory,results_path))
    t1.start()
    print("FINISHEDDD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----")
    return "ok" , 200

@app.route("/data/<name>" , methods = ["GET"])
def Get_City_Dates(name):
    Dates = []
    for i in mongo.db.shapeFile_classified.find({"name" : name}).sort([("Date" , pymongo.ASCENDING)]):
        Dates.append(i["Date"])
    return flask.jsonify(Dates)


@app.route("/date/all" , methods = ["GET"])
def Get_all_Dates():
    Dates = {}
    for i in mongo.db.shapeFile_classified.find().sort([("Date" , pymongo.ASCENDING)]):
        if(i["name"] in Dates):
            Dates[i["name"]].append(i["Date"])
        else:
            Dates[i["name"]] = []
            Dates[i["name"]].append(i["Date"])
        
    return flask.jsonify(Dates)

@app.route("/governorate/all" , methods = ["GET"])
def get_all_governates():

    names = []
    try:
        for i in mongo.db.shapeFile_original.find().sort([("name" ,pymongo.ASCENDING )]):

            names.append({"name" : i["name"] , "cord" :  i["cord"] , "type" : i["Type"]})
    except:
        return "internal server error" , 500

    
    return flask.jsonify(names)


@app.route("/governorate/<name>" , methods = ["GET"])
def get_Governate_data(name):
    data = None
    for i in mongo.db.shapeFile_original.find({"name" : name}).sort([("Date" ,pymongo.ASCENDING )]):
        data = {"name" : i["name"] , "cord" :  i["cord"] , "type" : i["Type"]}
        break
    return flask.jsonify(data) , 200

