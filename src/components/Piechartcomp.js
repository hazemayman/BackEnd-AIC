import React, { useState, useEffect } from "react";
import Chart from "react-apexcharts";
import { MediaQuery } from "react-responsive";
import { words } from "../helpers/lang";

function Piechartcomp({ lang, resources }) {
  let [labels, setLabels] = useState([
    words.ResourcesLegend.blue[lang],
    words.ResourcesLegend.green[lang],
    words.ResourcesLegend.yellow[lang],
    words.ResourcesLegend.magenta[lang],
    words.ResourcesLegend.purple[lang],
  ]);
  const [series, setSeries] = useState([
    resources.aqua,
    resources.agriculture_land,
    resources["sand-rocks"],
    resources["urban-land"],
    resources.road,
  ]);

  
  useEffect(() => {
    setLabels([
      words.ResourcesLegend.blue[lang],
      words.ResourcesLegend.green[lang],
      words.ResourcesLegend.yellow[lang],
      words.ResourcesLegend.magenta[lang],
      words.ResourcesLegend.purple[lang],
    ]);

    setSeries([
      resources.aqua,
      resources.agriculture_land,
      resources["sand-rocks"],
      resources["urban-land"],
      resources.road,
    ]);
  }, [lang, resources]);

  const [options, setOptions] = useState({
    series: series,
    labels: labels,
  });
  console.log()
  //responsivness
  return (
    <Chart
      options={options}
      series={series}
      categories={labels}
      type="donut"
      width="350"
      style={{ float: "right", display: "flex", color: "white" }}
      chartOptions={{ labels: labels }}
    />
  );
}

export default Piechartcomp;
