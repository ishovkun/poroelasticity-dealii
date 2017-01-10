// #include <right_hand_side.h>

// namespace RightHandSide {
//   using namespace dealii;
//   // ----------------------- BodyForces class --------------------------------
//   // template <int dim>
//   // BodyForces<dim>::BodyForces (const int direction)
//   //   :
//   //   Function<dim> (dim)
//   // {}

//   // template <int dim>
//   // inline
//   // void
//   // BodyForces<dim>::vector_value(Vector<double>   &values) const {
//   //   Assert(values.size() == dim,
//   //           ExcDimensionMismatch (values.size(), dim));
//   //   Assert(dim == 2, ExcNotImplemented());

//   //   values(0) = 0;
//   //   values(1) = 0;
//   // }

//   template <int dim>
//   // void
//   // BodyForces<dim>::vector_value_list(const std::vector<Point<dim> > &points,
//   //                                   std::vector<Vector<double> >   &value_list
//   //                                   ) const {
//   //   Assert (value_list.size() == points.size(),
//   //           ExcDimensionMismatch (value_list.size(), points.size()));
//   //   const unsigned int n_points = points.size();
//   //   for (unsigned int p=0; p < n_points; ++p)
//   //     BodyForces<dim>::vector_value(value_list[p]);
//   // }

//   // ----------------------- SinglePhaseSource -------------------------
//   template <int dim>
//   SinglePhaseWell<dim>::SinglePhaseWell(double r) :
//     Function<dim>(),
//     r_well(r)
//   {
//     // r_well = r;
//   }

//   template<int dim>
//   void SinglePhaseWell<dim>::setRate(double rate)
//   {
//     flow_rate = rate;
//   }

//   template<int dim>
//   double SinglePhaseWell<dim>::value(const Point<dim> &p,
//                                      const unsigned int component) const
//   {
//     Assert (component == 0, ExcInternalError());
//     Assert (dim == 2, ExcNotImplemented());

//     double r_squared = p[0]*p[0] + p[1]*p[1];
//     if (r_squared <= r_well*r_well)
//       return flow_rate;
//     else
//       return 0;
//   }

//   // ----------------------- FlowGravityclass -------------------------

// } //end of namespace
