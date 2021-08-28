import React, { useState, useEffect } from "react";
import { Dropdown } from "react-bootstrap";
import { words } from "../helpers/lang";
import { urls } from "../helpers/urls";

function Datecomp({ lang, dates }) {
  return (
    <Dropdown>
      <Dropdown.Toggle variant="dark" id="dropdown-basic">
        {words.DateDropDownTitle[lang]}
      </Dropdown.Toggle>
      <Dropdown.Menu style={{ overflowY: "scroll", maxHeight: "300px" }}>
        {dates.map((date) => {
          return <Dropdown.Item href={"#"}>{date}</Dropdown.Item>;
        })}
      </Dropdown.Menu>
    </Dropdown>
  );
}

export default Datecomp;
