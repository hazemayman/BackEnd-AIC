import React, { useState } from "react";
import { Dropdown } from "react-bootstrap";
import { words } from "../helpers/lang";
function CItyDropDownComp({ cities, lang }) {
  return (
    <Dropdown>
      <Dropdown.Toggle variant="Success" id="dropdown-basic">
        {words.CityDropDownTitle[lang]}
      </Dropdown.Toggle>
      <Dropdown.Menu style={{ overflowY: "scroll", maxHeight: "300px" }}>
        {cities.map((city) => {
          return <Dropdown.Item href={"#"}>{city}</Dropdown.Item>;
        })}
      </Dropdown.Menu>
    </Dropdown>
  );
}

export default CItyDropDownComp;
