cl1 = 1;
xSize = 10;
ySize = 10;

r_well = 0.1;
well_element_size = 0.02;

// Outer domain points
Point(1) = {-xSize/2, -ySize/2, 0, 1};
Point(2) = {+xSize/2, -ySize/2, 0, 1};
Point(3) = {+xSize/2, +ySize/2, 0, 1};
Point(4) = {-xSize/2, +ySize/2, 0, 1};

// wellbore points
Point(5) = {r_well, 0, 0, well_element_size};
Point(6) = {0, r_well, 0, well_element_size};
Point(7) = {-r_well, 0, 0, well_element_size};
Point(8) = {0, -r_well, 0, well_element_size};
Point(9) = {0, 0, 0, 1};

// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// wellbore line
Ellipse(5) = {5, 9, 6, 6};
Ellipse(6) = {6, 9, 7, 7};
Ellipse(7) = {7, 9, 8, 8};
Ellipse(8) = {8, 9, 5, 5};

// loops of the outside and the two cutouts
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};
Plane Surface(11) = {9, 10};

// these define the boundary indicators in deal.II:
Physical Line(0) = {1}; // Bottom
Physical Line(1) = {2}; // Right
Physical Line(2) = {3}; // Top
Physical Line(3) = {4}; // Left
Physical Line(4) = {5, 6, 7, 8}; // Well

// you need the physical surface, because that is what deal.II reads in
Physical Surface(12) = {11};

// some parameters for the meshing:
Mesh.Algorithm = 8;
// Mesh.CharacteristicLengthFactor = 0.09;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 20;
Mesh.RecombineAll = 1; // to get quadrelaterals
// Mesh.MshFileVersion = 1
Show "*";


///