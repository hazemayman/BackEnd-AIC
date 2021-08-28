import React, { useState , useEffect } from "react";
import {
  Offcanvas,
  Button,
  Dropdown,
  Row,
  Col,
  Image,
  Alert,
  Figure,
} from "react-bootstrap";
// import logo from "../helpers/files/abosombol.png";
import Piechartcomp from "./Piechartcomp";
import Govdropdowncomp from "./GovDropDownComp";
import Datecomp from "./Datecomp";
import Urbandropdowncomp from "./Urbanselectioncomp";
import Navcomp from "./Navcomp";
import { words, langs } from "../helpers/lang";
import { govs, dates, cities } from "../helpers/dummy";
import Resourcescomp from "./Resourcescomp";

const LangDropdown = ({ setStateFunc, lang }) => {
  return (
    <Dropdown>
      <Dropdown.Toggle
        variant="Success"
        id="dropdown-basic"
        style={{ color: "white" }}
      >
        {lang}
      </Dropdown.Toggle>
      <Dropdown.Menu>
        {langs.map((lang) => {
          return (
            <Dropdown.Item onClick={() => setStateFunc(lang)} href={"#"}>
              {lang}
            </Dropdown.Item>
          );
        })}
      </Dropdown.Menu>
    </Dropdown>
  );
};

function OffCanvascomp({ lang, setLangStateFunc, resources, resourcesTitle ,imgpath, showCanvas}) {
  const [show, setShow] = useState(true);
  const [imgURI , setImgURI] = useState(imgpath)
  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  useEffect(() => {
    setImgURI(imgpath)
  }, [imgpath]); 
  return (
    <>
      {/* <Button variant="primary" onClick={handleShow} className="me-2"></Button> */}
      <Offcanvas
        show={showCanvas}
        onHide={handleClose}
        placement={"end"}
        scroll={true}
        backdrop={false}
        style={{
          backgroundColor: "#6A7A7F",
          color: "white",
        }}
      >
        <Navcomp lang={lang} />
        <Offcanvas.Header
          /* closeButton */
          style={{ paddingTop: "0", paddingBottom: "2px" }}
        >
          <Offcanvas.Title style={{ display: "flex", color: "white" }}>
            <strong>{words.OffCanvasHeader[lang]}</strong>
          </Offcanvas.Title>
          <LangDropdown setStateFunc={setLangStateFunc} lang={lang} />
        </Offcanvas.Header>
        <Offcanvas.Body>
          {/* ID: A1 */}
          <Piechartcomp lang={lang} resources={resources} />
          {/* <Image src="#" rounded style={{ backgroundColor: "black" }} /> */}
          {/* <Button variant="link">Show Info</Button> */}
          <div
            style={{
              textAlign: "center",
            }}
          >
            <h6>{resourcesTitle}</h6>
            <Resourcescomp resources={resources} lang={lang} />
          </div>
          <Figure
            style={{
              textAlign: "center",
            }}
          >

            <Figure.Image width={"100%"}src={"http://127.0.0.1:5000/" + imgURI } />
          </Figure>
          {/* {console.log("")} */}
        </Offcanvas.Body>
        {/* <div
          className="d-grid gap-2"
          style={{ position: "absolute", bottom: "10px", marginLeft: "1rem" }}
        >
          <Button variant="light" size="sm">
            {words.OffCanvasButton[lang]}
          </Button>
        </div> */}
      </Offcanvas>
    </>
  );
}

export default OffCanvascomp;
// ID: A1
// {/* <Row>
//   <Col>
//     {/* Gov Dropdown */}
//     <Govdropdowncomp govs={govs} lang={lang} />
//     {/* City Dropdown */}
//   </Col>
//   {/* <Col>
//               <Citydropdowncomp cities={cities} lang={lang} />
//             </Col> */}
//   <Col>
//     <Urbandropdowncomp />
//   </Col>
//   <Col>
//     <Datecomp values={dates} lang={lang} />
//   </Col>
// </Row>; */}
