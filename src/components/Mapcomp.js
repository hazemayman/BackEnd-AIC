import React, { useEffect, useState, memo } from "react";
import DeckGL from "@deck.gl/react";
import { PolygonLayer } from "@deck.gl/layers";
import { StaticMap, FlyToInterpolator } from "react-map-gl";
import { urls } from "../helpers/urls";
import { Container, Row, Col , Button } from "react-bootstrap";
import Govdropdowncomp from "./GovDropDownComp";
import Datecomp from "./Datecomp";
import { words } from "../helpers/lang";
import { govs, dates, cities } from "../helpers/dummy";

// Viewport settings

const MAPBOX_ACCESS_TOKEN =
  "pk.eyJ1IjoiZG9uem9tYTA5IiwiYSI6ImNrcno0djVqNjAwMGEyd3BjanVuY2hqMGIifQ.sOkp8Uufulj_dbEEpHrh3w";
// Data to be used by the LineLayer

// const transitionInterpolator = new Linear(['bearing']);
const LONGITUDE_RANGE = [22.8357675, 32.8357675];
const LATITUDE_RANGE = [22.5, 31.1956597];

const raw_view_state = {
  longitude: 31.2,
  latitude: 27.8025,
  zoom: 6,
  pitch: 10,
  bearing: 0,
  transitionDuration: 900,
  transitionInterpolator: new FlyToInterpolator(20),
  minZoom: 6,
  maxZoom: 14,
};

const myfunction = (data) => {
  return [0, 0, 0];
};

// DeckGL react component
const MapComponent = memo(({ lang, dates, govs, data, fetchSelection , Country , toggleOffCanvasFunc}) => {
  const [hoverInfo, setHoverInfo] = useState({});
  const [view_state, set_view_state] = useState(raw_view_state);
  const [polygon_data, set_polygon_data] = useState([{}]);
  const [hoveredObject, setHoveredObject] = useState(-1);
  const [clickedObject, setClickedObject] = useState({});

  //   const rotateCamera = useCallback(() => {
  //     set_view_state(viewState => ({
  //       ...view_state,
  //       bearing: viewState.bearing + 120,
  //       transitionDuration: 1000,
  //       transitionInterpolator,
  //       onTransitionEnd: rotateCamera
  //     }))
  //   }, []);
  const getIndex = (name) => {
    for (const i in polygon_data) {
      if (polygon_data[i]["name"] == name) {
        return i;
      }
    }
  };
  useEffect(() => {
    set_polygon_data(data)
  }, [data]); 
  // useEffect(() => {
  //   if(Country != ""){
  //     const index = getIndex(Country);
  //     const newData = [...polygon_data];
  //     if (newData[index] != undefined) {
  //       console.log(newData[index])
  //     }
  //   }
    
  // }, [Country]); 
  useEffect(() => {
    if (hoverInfo.index != -1) {
      const index = hoverInfo.index;

      const newData = [...polygon_data];
      if (newData[index] != undefined) {
        if (index != hoveredObject && hoveredObject != -1) {
          newData[hoveredObject].lineWeight = 1;
          newData[hoveredObject].color = [240, 240, 240, 20];
        }
        newData[index].lineWeight = 10;
        newData[index].color = [100, 40, 40, 15];
        setHoveredObject(index);
        set_polygon_data(newData);
      }
    } else {
      const newData = [...polygon_data];
      if (newData[hoveredObject] != undefined) {
        newData[hoveredObject].lineWeight = 1;
        newData[hoveredObject].color = [240, 240, 240, 20];
        set_polygon_data(newData);
      }
    }
  }, [hoverInfo]);

  useEffect(() => {
    const index = clickedObject.index;
    const newView = Object.assign({}, raw_view_state);
    if (polygon_data[index] != undefined) {
      let country = clickedObject['object']['name'];
      fetchSelection(country);
      if (view_state.zoom < 8) {
        newView.longitude = clickedObject.coordinate[0];
        newView.latitude = clickedObject.coordinate[1];
        newView.zoom = 10.5;
        set_view_state(newView);
      }
    }
  }, [clickedObject]);
  const layer = new PolygonLayer({
    id: "PolygonLayer",
    data: polygon_data,

    /* props from PolygonLayer class */

    elevationScale: 1,
    extruded: false,
    filled: true,
    getLineColor: [200, 80, 80],
    getLineWidth: (d) => 100,
    getPolygon: (d) => d.contour,
    // lineJointRounded: true,
    // lineMiterLimit: 4,
    lineWidthMaxPixels: Number.MAX_SAFE_INTEGER,
    lineWidthMinPixels: 1,
    // lineWidthScale: 1,
    // lineWidthUnits: 'meters',
    // material: true,
    stroked: true,
    wireframe: true,

    /* props inherited from Layer class */

    // autoHighlight: false,
    // coordinateOrigin: [0, 0, 0],
    // coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    // highlightColor: [0, 0, 128, 128],
    // modelMatrix: null,
    // opacity: 1,
    pickable: true,
    // visible: true,
    // wrapLongitude: false,
  });

  const getLineWeight = (data) => {
    return data.lineWeight;
  };
  return (
    <DeckGL
      viewState={view_state}
      onViewStateChange={e => set_view_state(e.viewState)}
      // onLoad={rotateCamera}
      controller={true}
      onViewStateChange={({ viewState }) => {
        viewState.longitude = Math.min(
          LONGITUDE_RANGE[1],
          Math.max(LONGITUDE_RANGE[0], viewState.longitude)
        );
        viewState.latitude = Math.min(
          LATITUDE_RANGE[1],
          Math.max(LATITUDE_RANGE[0], viewState.latitude)
        );
        set_view_state(viewState);
      }
    }
    >
      <PolygonLayer
        id="PolygonLayer"
        data={polygon_data}
        elevationScale={1}
        extruded={false}
        getLineWidth={(d) => getLineWeight(d)}
        filled={true}
        getFillColor={(d) => d.color}
        pickable={true}
        getLineColor={(d) => myfunction(d)}
        getPolygon={(d) => d.contour}
        onHover={(d) => setHoverInfo(d)}
        onClick={(d) => setClickedObject(d)}
      ></PolygonLayer>
      <StaticMap mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN}></StaticMap>
      <div
        style={{
          display: "flex",
          float: "left",
          margin: "1rem",
          padding: "0",
          position: "static",
          justifyContent: "space-around",
          width: "25%",
        }}
      >
        <Govdropdowncomp
          lang={lang}
          govs={govs}
          fetchSelection={fetchSelection}
        />
        <Datecomp lang={lang} dates={dates} />
        <Button onClick={() => toggleOffCanvasFunc()}>
            {words.DashboardButton[lang]}
          </Button>
      </div>
      
    </DeckGL>
  );
});

export default MapComponent;
