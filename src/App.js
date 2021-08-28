import "./css/App.css";
import "./css/Table.css";
/* import Navcomp from "./components/Navcomp"; */

import SummaryTable from "./components/SummaryTable";
import Mapcomp from "./components/Mapcomp";
/* import { govs, dates, cities } from "./helpers/dummy"; */
import { Container, Row, Col, Dropdown } from "react-bootstrap";
import OffCanvascomp from "./components/OffCanvascomp";
import React, { useState, useEffect } from "react";
import Loadingcomponent from "./components/Loadingcomponent";
import { urls } from "./helpers/urls";

function App() {
  //@TODO language config
  const [lang, setLang] = useState("en"); //responsible for the language dropdown state
  const [dates, setDates] = useState([]); // responsible for the Dates dropdown state
  const [govs, setGovs] = useState([]); //responsible for the governorate drop down state
  const [resources, setResources] = useState({
    agriculture_land: 0,
    aqua: 0,
    road: 0,
    "sand-rocks": 0,
    trees: 0,
    unknown: 0,
    "urban-land": 0,
  }); //responsible for the  resources cards state
  const [loading, setLoading] = useState(true); //responsible for the loading state
  const [error, setError] = useState(false); //responsibel for the error state
  const [mapdata, setmapData] = useState([{}]);
  const [resourcesTitle, setResourcesTitle] = useState("All Governorates");
  const [countryName , seetCountryName ] = useState("")
  const [imgURI , setImgURI] = useState("static/imgs/mash_abohomos.png")
  const [showCanvas, setShowCanvas] = useState(true); //switches the canvas on/off

  ///----- Functions -----///
  // fetches the selection of the governorate dropdown
  function toggleOffCanvas() {
    setShowCanvas(!showCanvas);
  }
  async function fetchGovSelection(selection) {
    setLoading(true);
    setResourcesTitle(selection);
    seetCountryName(selection)
    await fetch(urls.server + "/resources/" + selection)
      .then((res) => res.json())
      .then((data) => {
        setResources(data.resource);
        setLoading(false);
        if(data.imgURI != undefined){
          setImgURI(data.imgURI)
        }else{
          setImgURI("")
        }
      });
  }

  ///----- UseEffects -----///
  // Get all needed data for the startup
  useEffect(() => {
    //fetches the governorates data
    const fetchGovs = async () => {
      fetch(urls.server + "/date/all")
        .then((res) => res.json())
        .then((data) => {
          setGovs(Object.keys(data));
        });
    };

    //fetches the date data
    const fetchDates = async () => {
      fetch(urls.server + "/date/all")
        .then((res) => res.json())
        .then((data) => {
          let temp = [];
          for (const gov in data) {
            temp.push(data[gov]);
          }
          setDates(temp);
        });
    };

    //fetches aggregated resources for all governorates
    const fetchAllAggregatedResources = async () => {
      fetch(urls.server + "/resources/all")
        .then((res) => res.json())
        .then((data) => {
          setResources(data);
        });
    };

    // Actual calls
    fetchGovs();
    fetchDates();
    fetchAllAggregatedResources();
    setLoading(false);
  }, []);

  //fetches map data
  useEffect(() => {
    const getCountries = async () => {
      await fetch(urls.server + "/governorate/all")
        .then((response) => response.json())
        .then((data) => {
          const coming_data = [];
          for (const i in data) {
            if (data[i]["type"] == "MultiPolygon") {
              continue;
            }
            coming_data.push({
              contour: data[i]["cord"],
              lineWeight: 1,
              name: data[i]["name"],
              type: data[i]["type"],
              color: [240, 240, 240, 20],
            });
          }
          console.log(coming_data)
          return coming_data;
        })
        .then((data) => {
          setmapData(data);
          console.log(data);
        });
    };
    getCountries();
  }, []);

  return (
    <div>
      <header>
        {/* Navbar [drop down to select gov && dropdown to select region && dropdown to select date*/}
        {/* <Navcomp style={{ position: "absolute" }} /> */}
      </header>
      <body>
        {/* Grid */}

        <div
          style={{
            margin: "1rem 0",
            justifyContent: "space-between",
          }}
        >
          {/* <Container>
            <div
              style={{
                display: "flex",
                justifyContent: "space-evenly",
                float: "left",
                margin: "0.2rem 0",
              }}
            >
              <Datecomp values={dates} lang={lang} />

              <Urbandropdowncomp />
            </div>
            </Container>*/}
          {/* First Row  */}
          <Row>
            <Col s={12} md={7}>
              <Mapcomp
                toggleOffCanvasFunc = {toggleOffCanvas}
                lang={lang}
                dates={dates}
                govs={govs}
                fetchSelection={fetchGovSelection}
                setLoadingFunc={setLoading}
                setResourcesTitle={setResourcesTitle}
                data={mapdata}
                Country = {countryName}
              />
            </Col>
            <Col s={12} md={5}>
              <OffCanvascomp
                showCanvas = {showCanvas}
                imgpath = {imgURI}
                setLangStateFunc={setLang}
                lang={lang}
                resources={resources}
                setLoadingFunc={setLoading}
                resourcesTitle={resourcesTitle}
                
              />
            </Col>
          </Row>
        </div>
      </body>
    </div>
  );
}

export default App;
