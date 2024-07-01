
======
geom2d
======

A simple pure Python 2D geometry package that includes
point, line, arc, ellipse, cubic Bezier curves,
and miscellaneous implementations of computational geometry solutions
such as Voronoi diagrams, convex hull, biarc approximation of Bezier curves,
polygon intersection, etc.

All objects (points, lines, etc) are implemented as immutable tuples.

* Documentation: https://utlco.github.io/utl-geom2d
* Repository: https://github.com/utlco/utl-geom2d
* License: LGPL v3
* :ref:`modindex`


.. toctree::
    :maxdepth: 2

    api


Credits
-------

I am very grateful for the work of these people.

Joseph O'Rourke: computational geometry
https://www.science.smith.edu/~jorourke/index.html

Paul Bourke: computational geometry
http://paulbourke.net

Eric Haines: computational geometry
http://erich.realtimerendering.com/

W. Randolph Franklin: short and speedy point in polygon algorithm
https://wrfranklin.org/Research/Short_Notes/pnpoly.html

Pomax: great primer on Bezier curves
https://pomax.github.io/bezierinfo

Adrian Colomitchi: finding inflection points of a cubic Bezier curve
(his website seems to have disappeared but is still available via the wayback machine:
http://web.archive.org/web/20220129063812/https://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html

Earl Boebert: Computing the arc length of a cubic Bezier
http://steve.hollasch.net/cgindex/curves/cbezarclen.html

And many others.

If you find code from your open source project without attribution please email me
and I will fix it.


Apologies/caveats/etc.
----------------------

Testing is artisanal at best.

Any math errors are my fault alone and,
while the library is fast enough for the Inkscape extensions I write,
there may be obvious optimizations that I've overlooked or purposely ignored for
the sake of clarity (to me).


