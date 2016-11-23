#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace RightHandSide {
  using namespace dealii;
  //
  /*----------------------- BodyForces --------------------------------
    This is a class that imposes gravity on deformation system RHS
   */
  template <int dim>
  class BodyForces :  public Function<dim>
    {
    public:
      BodyForces (const int direction);

      virtual void vector_value (Vector<double> &values) const;
      virtual void vector_value_list (const std::vector< Point<dim> > &points,
                                      std::vector<Vector<double> > &value_list) const;
    private:
      int direction;
    };

  //
  /*----------------------- SinglePhaseSource --------------------------------
    This is a class that imposes wells in single phase flow model
  */
  template <int dim>
    class SinglePhaseSource :  public Function<dim>
    {
    public:
      SinglePhaseSource(double r);
      virtual void setRate(double rate);

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;
    private:
      double r_well, flow_rate;
    };

} // end of namespace
