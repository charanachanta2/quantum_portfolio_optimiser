const pptxgen=require("pptxgenjs"); const fs=require("fs");
const J=JSON.parse(fs.readFileSync("steps.json")); const S=J.steps, TAG=J.data_tag;
const sizeOf=require("child_process").execSync;
const pres=new pptxgen(); pres.layout="LAYOUT_WIDE"; // 13.33 x 7.5
pres.title="QPO Code Walkthrough"; pres.theme={headFontFace:"Cambria",bodyFontFace:"Calibri"};
const NAVY="1B2A49",TEAL="139A8C",MINT="E4F4F1",INK="1F2937",GRY="6B7280";
function dims(p){const o=sizeOf(`python3 -c "from PIL import Image;print(*Image.open('${p}').size)"`).toString().trim().split(" ").map(Number);return o;}
// title slide
let s=pres.addSlide(); s.background={color:NAVY};
s.addText("Fault-Tolerant Quantum Portfolio Optimizer",{x:0.8,y:2.0,w:11.5,h:1.5,fontFace:"Cambria",fontSize:44,bold:true,color:"FFFFFF",isTextBox:true,margin:0});
s.addText("Step-by-step code walkthrough, with figures generated from the project's own code",{x:0.8,y:3.7,w:11,h:0.8,fontSize:22,color:"7FD6CB",isTextBox:true,margin:0});
s.addText("Numbers: "+TAG+"  |  illustrative 12-qubit instance  |  "+S.length+" steps",{x:0.8,y:6.2,w:11,h:0.4,fontSize:14,color:"B8C2D6",isTextBox:true,margin:0});
S.forEach((st,i)=>{
  const sl=pres.addSlide(); sl.background={color:"FFFFFF"};
  sl.addText(st.title,{x:0.6,y:0.35,w:12.1,h:0.8,fontFace:"Cambria",fontSize:34,bold:true,color:NAVY,isTextBox:true,margin:0});
  sl.addText("Where: "+st.sub,{x:0.6,y:1.15,w:12.1,h:0.4,fontSize:16,color:TEAL,isTextBox:true,margin:0});
  const hasImg=!!st.img;
  const bw=hasImg?5.4:12.1;
  const items=st.bullets.map((b,k)=>({text:b,options:{bullet:true,breakLine:k<st.bullets.length-1,paraSpaceAfter:8}}));
  sl.addText(items,{x:0.6,y:1.8,w:bw,h:4.3,fontSize:hasImg?15:18,color:INK,valign:"top",isTextBox:true,margin:0});
  if(hasImg){
    const [w,h]=dims(st.img); const maxW=6.6,maxH=4.3; let W=maxW,H=W*h/w; if(H>maxH){H=maxH;W=H*w/h;}
    sl.addImage({path:st.img,x:6.3+(maxW-W)/2,y:1.8+(maxH-H)/2,w:W,h:H});
  }
  sl.addShape(pres.ShapeType.roundRect,{x:0.6,y:6.35,w:12.1,h:0.7,fill:{color:MINT},line:{color:MINT},rectRadius:0.08});
  sl.addText([{text:"Takeaway: ",options:{bold:true,color:NAVY}},{text:st.take,options:{color:INK}}],{x:0.8,y:6.35,w:11.7,h:0.7,fontSize:15,valign:"middle",isTextBox:true,margin:0});
  sl.addText(st.cap,{x:6.3,y:6.08,w:6.4,h:0.25,fontSize:10,italic:true,color:GRY,isTextBox:true,margin:0,align:"right"});
  sl.addText(String(i+1),{x:12.3,y:0.3,w:0.5,h:0.4,fontSize:12,color:GRY,align:"right",isTextBox:true,margin:0});
  sl.addNotes(st.bullets.join(" "));
});
// closing
s=pres.addSlide(); s.background={color:NAVY};
s.addText("Where the project stands",{x:0.8,y:0.7,w:11,h:0.9,fontFace:"Cambria",fontSize:38,bold:true,color:"FFFFFF",isTextBox:true,margin:0});
s.addText([
 {text:J.status[0],options:{bullet:true,breakLine:true,paraSpaceAfter:10}},
 {text:J.status[1],options:{bullet:true,breakLine:true,paraSpaceAfter:10}},
 {text:J.status[2],options:{bullet:true}}
 ],
 {x:0.8,y:2.0,w:11.5,h:4,fontSize:20,color:"E8EEF7",valign:"top",isTextBox:true,margin:0});
pres.writeFile({fileName:"QPO_Code_Walkthrough.pptx"}).then(()=>console.log("ok"));
