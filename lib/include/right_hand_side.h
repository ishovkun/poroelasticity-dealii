#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace RightHandSide {
  using namespace dealii;

  /*----------------------- BodyForces --------------------------------
    This is a class that imposes gravity on deformation system RHS
   */
  template <int dim>
  class BodyForces :  public Function<dim>
    {
    public:
      BodyForces (int d, double rho);

      virtual void vector_value (Vector<double> &values) const;
      virtual void vector_value_list (const std::vector< Point<dim> > &points,
                                      std::vector<Vector<double> > &value_list) const;
    private:
      const int direction;
      double density;
    };

  /*----------------------- SinglePhaseSource --------------------------------
    This is a class that imposes wells in single phase flow model
  */
  template <int dim>
    class SinglePhaseWell : public Function<dim>
    {
    public:
      SinglePhaseWell(double r);
      /* ~SinglePhaseWell(); */
      virtual void setRate(double rate);

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;
    private:
      double r_well, flow_rate=1e6;
    };

  // ----------------------- IMPLEMENTATION --------------------------------


  // ----------------------- BodyForces class --------------------------------
  template <int dim>
    BodyForces<dim>::BodyForces (int d, double rho)
    :
    Function<dim> (dim),
    direction(d),
    density(rho)
  {}

  template <int dim>
  inline
  void
  BodyForces<dim>::vector_value(Vector<double>   &values) const {
    Assert(values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    /* Assert(dim == 2, ExcNotImplemented()); */

    for (int i=0; i<dim; i++){
      values(i) = 0;
    }

    if (direction <= dim){
      values(direction) = -9.81*density;
    }
  }

  template <int dim>
  void
  BodyForces<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list
                                    ) const {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));
    const unsigned int n_points = points.size();
    for (unsigned int p=0; p < n_points; ++p)
      BodyForces<dim>::vector_value(value_list[p]);
  }

  // ----------------------- SinglePhaseSource -------------------------
  template <int dim>
  SinglePhaseWell<dim>::SinglePhaseWell(double r) :
    Function<dim>(),
    r_well(r)
  {
    // r_well = r;
  }

  template<int dim>
  void SinglePhaseWell<dim>::setRate(double rate)
  {
    flow_rate = rate;
  }

  template<int dim>
  double SinglePhaseWell<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    Assert (dim == 2, ExcNotImplemented());

    double r_squared = p[0]*p[0] + p[1]*p[1];
    if (r_squared <= r_well*r_well)
      return flow_rate;
    else
      return 0;
  }

  // ----------------------- FlowGravityclass -------------------------

} // end of namespace
