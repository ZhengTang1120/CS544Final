var rects = [];
var maxr = 0;
var minr = 0;
var absmax = 0;
for (var i=0;i<words.length;i++){
	for (var j=0;j<words[i].length;j++){
		maxr = Math.max(maxr, Math.max(...words[i][j].response));
		minr = Math.min(minr, Math.min(...words[i][j].response));
		absmax = Math.max(absmax, Math.max(...words[i][j].response.map(Math.abs)));
	}
}
var a = Math.min(Math.abs(minr), maxr);
var lScale = d3.scaleLinear().domain([0, a]).range([50,100]);
var hScale = d3.scaleLinear().domain([-a, a]).range([-100,100]);
function setRect(){
	for (var i=0;i<hidden_units.length;i++){
		var k = Math.ceil(hidden_units[i].length/2)
		for (var j=0;j<hidden_units[i].length;j++){
			var color = d3.lab(50,0,0);
			if (j < k){
				subrect = {x1:5+j*25,x2:25+j*25,y1:5,y2:25,name:hidden_units[i][j], color:color};
			}else{
				subrect = {x1:5+(j-k)*25,x2:25+(j-k)*25,y1:30,y2:50,name:hidden_units[i][j], color:color};
			}
			rects[hidden_units[i][j]] = subrect;
		}
	}
}

function setAttrs(sel) {
    // WRITE THIS PART.
    sel.attr("width", function(id) { 
    	rect = rects[id];
        return Math.max(1, rect.x2 - rect.x1);
    }).attr("height", function(id) {
    	rect = rects[id];
        return Math.max(1, rect.y2 - rect.y1);
    }).attr("x", function(id) {
    	rect = rects[id];
        return rect.x1;
    }).attr("y", function(id) {
    	rect = rects[id];
        return rect.y1;
    }).attr("fill", function(id) {
    	rect = rects[id];
        return rect.color;
    }).attr("stroke", function() {
        return d3.hcl(50, 50, 0);
    }).attr("title", function(id) {
    	rect = rects[id];
        return rect.name;
    })
    .attr("class", "hidden_units");
}

setRect();
var gs = d3.select("#hidden")
		.selectAll("div")
		.data(hidden_units)
		.enter()
		.append("div")
		.attr("class", "hidden_frame")
		.append("svg")
		.selectAll("rect")
		.data(function(d){
			return d;
		})
		.enter();

gs.append("rect").call(setAttrs);
var pre;
var gs2 = d3.select("#word")
	        .selectAll("div")
	        .data(words)
	        .enter()
	        .append("div")
	        .attr("id", function(d, i){
	        	return "word_cluster_"+i.toString();
	        })

	        
Highcharts.seriesTypes.wordcloud.prototype.deriveFontSize = function (relativeWeight) {
   var maxFontSize = 10;
  // Will return a fontSize between 0px and 25px.
  return Math.floor(maxFontSize * relativeWeight);
};

var width_w = window.innerWidth;
var height_w = window.innerHeight;

for(var i=0; i<words.length; i++){
	Highcharts.chart("word_cluster_"+i.toString(), {
		chart: {
			borderColor: '#EBBA95',
        	borderWidth: 2,
	        height: 100,
	        width: width_w/3
	    },
		plotOptions: {
	        series: {
	        	allowPointSelect: true,
		    	colors: ['blue'],
		        rotation: {
		            from: 0,
		            to: 0,
		            orientations: 1
		        },
		        events:{
		        	click: function(){
		        		d3.selectAll(".active").attr("class", "inactive");
						d3.selectAll("#edges_"+eval(this.name)).attr("class", "active");
		        	}
		        },
	        	point: {
		    		events: {
			    		click: function (event) {
			    			if (pre){
			    				pre.update({color: "blue"});
			    			}
			                this.update({color: "red"});
			                pre = this;
			                for (var i=0;i<rects.length;i++){
						    	rects[i].color = d3.lab(lScale(Math.abs(this.response[i])),0,hScale(this.response[i]));
						    }
							d3.selectAll(".hidden_units").transition().duration(100).call(setAttrs);
							Highcharts.chart('chart', {
							    title: {
							        text: null
							    },
							    yAxis: {
							        title: {
							            text: 'response'
							        }
							    },
							    xAxis: {
							    	title: {
							    		text: 'dimension'
							    	}
							    },
							    legend: {
							        layout: 'vertical',
							        align: 'right',
							        verticalAlign: 'middle'
							    },
							    series: [{
							    	name: this.name,
							        data: this.response
							    }]
							});		           
						}
			    	}
		    	}
	        }
	    },
	    series: [{
	        data: words[i],
	        type: 'wordcloud',
	        name: i.toString()
	    }],
	    title: {
	        text: null
	    },
	    tooltip: {
	    	enabled: false
	    }
	});
}

function setEdges(){
	var g = d3.select("#edges")
			.append("svg")
			.attr("width", width_w/3-10)
			.attr("height", height_w)
	for (var i=0;i<edges.length;i++){
		absmax = Math.max(...edges[i].map(Math.abs));
		var wScale = d3.scaleLinear().domain([0,absmax]).range([0,5]);
		for (var j=0;j<edges[i].length;j++){
			g.append("line").attr("x1",width_w/3-10).attr("y1",60+100*i)
							.attr("x2",0).attr("y2",37.5+55*j)
							.attr("stroke",d3.lab(lScale(Math.abs(edges[i][j])),0,hScale(edges[i][j])))
							.attr("stroke-width",wScale(Math.abs(edges[i][j])))
							.attr("class", "inactive")
							.attr("id", "edges_"+i.toString());
		}
	}
}
setEdges();