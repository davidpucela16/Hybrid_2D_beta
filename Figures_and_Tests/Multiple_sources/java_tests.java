/*
 * java_tests.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 3 2022, 16:49 by COMSOL 4.1.0.88. */
public class java_tests {

  public static void main(String[] args) {
    run();
  }

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model
         .modelPath("/home/pdavid/Bureau/SS/2D_cartesian/Updated_BCs/Figures_and_Tests/Multiple_sources");

    model.modelNode().create("mod1");

    model.geom().create("geom1", 2);

    model.mesh().create("mesh1", "geom1");

    model.physics().create("poeq", "PoissonEquation", "geom1");

    model.study().create("std1");
    model.study("std1").feature().create("stat", "Stationary");

    model.geom("geom1").lengthUnit("\u00b5m");
    model.geom("geom1").feature().create("sq1", "Square");
    model.geom("geom1").feature("sq1").set("base", "corner");
    model.geom("geom1").feature("sq1")
         .set("pos", new String[]{"-0.4", "-0.2"});
    model.geom("geom1").feature("sq1").set("l", "0.4");
    model.geom("geom1").run("sq1");
    model.geom("geom1").run();

    model.param().set("L", "240");
    model.param().descr("L", "");
    model.param().set("x_0", "0.45-0.06");
    model.param().descr("x_0", "");
    model.param().set("y_0", "0.02-0.03");
    model.param().descr("y_0", "");
    model.param().set("x_1", "0.24-0.06");
    model.param().descr("x_1", "");
    model.param().set("y_1", "0.17-0.03");
    model.param().descr("y_1", "");
    model.param().set("x_2", "0.6-0.06");
    model.param().descr("x_2", "");
    model.param().set("y_2", "0.23-0.03");
    model.param().descr("y_2", "");
    model.param().set("x_3", "0.23-0.06");
    model.param().descr("x_3", "");
    model.param().set("y_3", "0.27-0.03");
    model.param().descr("y_3", "");
    model.param().set("x_4", "0.55-0.06");
    model.param().descr("x_4", "");
    model.param().set("y_4", "0.33-0.03");
    model.param().descr("y_4", "");
    model.param().set("x_5", "1.02-0.06");
    model.param().descr("x_5", "");
    model.param().set("y_5", ".41-0.03");
    model.param().descr("y_5", "");
    model.param().set("x_6", "0.96-0.06");
    model.param().descr("x_6", ".");
    model.param().set("y_6", "0.43-0.03");
    model.param().descr("y_6", "");
    model.param().set("x_7", "0.27-0.06");
    model.param().descr("x_7", "");
    model.param().set("y_7", "0.6-0.03");
    model.param().descr("y_7", "");
    model.param().set("x_8", "0.53-0.06");
    model.param().descr("x_8", "");
    model.param().set("y_8", "0.65-0.03");
    model.param().descr("y_8", "");
    model.param().set("x_9", "0.59-0.06");
    model.param().descr("x_9", "");
    model.param().set("y_9", "0.62-0.03");
    model.param().descr("y_9", "");
    model.param().set("x_10", "0.67-0.06");
    model.param().descr("x_10", "");
    model.param().set("y_10", "0.69-0.03");
    model.param().descr("y_10", "");
    model.param().set("x_11", "0.13-0.06");
    model.param().descr("x_11", "");
    model.param().set("y_11", "0.75-0.03");
    model.param().descr("y_11", "");
    model.param().set("x_12", "0.15-0.06");
    model.param().descr("x_12", "");
    model.param().set("y_12", "0.93-0.03");
    model.param().descr("y_12", "");
    model.param().set("x_13", "0.2-0.06");
    model.param().descr("x_13", "");
    model.param().set("y_13", "0.87-0.03");
    model.param().descr("y_13", "");
    model.param().set("x_14", "0.28-0.06");
    model.param().descr("x_14", "");
    model.param().set("y_14", "0.98-0.03");
    model.param().descr("y_14", "");
    model.param().set("x_15", "0.8-0.06");
    model.param().descr("x_15", "");
    model.param().set("y_15", "0.85-0.03");
    model.param().descr("y_15", "");
    model.param().set("x_16", "0.83-0.06");
    model.param().descr("x_16", "");
    model.param().set("y_16", "0.92-0.03");
    model.param().descr("y_16", "");
    model.param().set("alpha", "50");
    model.param().descr("alpha", "");
    model.param().set("R", "L/alpha");
    model.param().descr("R", "");

    model.geom("geom1").feature("sq1").set("size", "L");
    model.geom("geom1").feature("sq1").setIndex("pos", "0", 0);
    model.geom("geom1").feature("sq1").setIndex("pos", "0", 1);
    model.geom("geom1").runAll();
    model.geom("geom1").feature().create("c1", "Circle");
    model.geom("geom1").feature("c1").set("base", "center");
    model.geom("geom1").feature("c1")
         .set("pos", new String[]{"0.45", "0.25"});
    model.geom("geom1").feature("c1").set("r", "0.1");
    model.geom("geom1").run("c1");
    model.geom("geom1").run();
    model.geom("geom1").run("c1");
    model.geom("geom1").feature().create("dif1", "Difference");
    model.geom("geom1").feature("dif1").selection("input")
         .set(new String[]{"sq1"});
    model.geom("geom1").feature("dif1").selection("input2")
         .set(new String[]{"c1"});
    model.geom("geom1").run("dif1");
    model.geom("geom1").run();
    model.geom("geom1").feature("c1").set("r", "R");
    model.geom("geom1").feature("c1").setIndex("pos", "45", 0);
    model.geom("geom1").feature("c1").setIndex("pos", "45", 1);
    model.geom("geom1").runAll();
    model.geom("geom1").run();

    model.physics("poeq").feature("peq1").set("f", 1, "0");
    model.physics("poeq").feature().create("flux1", "FluxBoundary", 1);

    model.param().set("K_0", "1");
    model.param().set("K_com", "K_0/(2*pi*R)");

    model.physics("poeq").feature("flux1").set("g", 1, "K_com");
    model.physics("poeq").feature("flux1").set("q", 1, "K_com");
    model.physics("poeq").feature().create("dir1", "DirichletBoundary", 1);
    model.physics("poeq").feature("dir1").selection().set(new int[]{1, 2});

    model.mesh("mesh1").run();

    model.sol().create("sol1");
    model.sol("sol1").study("std1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", 2);
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("surf1", "Surface");
    model.result("pg1").feature("surf1").set("expr", "u");
    model.result("pg1").feature("surf1").set("descr", "Dependent variable u");

    model.sol("sol1").runAll();

    model.result("pg1").set("windowtitle", "Graphics");
    model.result("pg1").run();

    model.physics("poeq").feature("flux1").selection()
         .set(new int[]{5, 6, 7, 8});

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result().numerical().create("gev1", "EvalGlobal");
    model.result().numerical().create("int1", "IntLine");
    model.result().numerical("int1").selection().set(new int[]{5, 6, 7, 8});
    model.result().numerical("int1").set("expr", "poeq.g_u");
    model.result().numerical("int1").set("descr", "Boundary flux/source");
    model.result().table().create("tbl1", "Table");
    model.result().table("tbl1").comments("Line Integration 1 (poeq.g_u)");
    model.result().numerical("int1").set("table", "tbl1");
    model.result().numerical("int1").setResult();

    model.geom("geom1").lengthUnit("m");

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result().table().create("tbl2", "Table");
    model.result().table("tbl2").comments("Line Integration 1 (poeq.g_u)");
    model.result().numerical("int1").set("table", "tbl2");
    model.result().numerical("int1").setResult();
    model.result().numerical("gev1").set("data", "none");
    model.result().dataset().create("int1", "Integral");
    model.result().dataset("int1").set("showlevel", "on");
    model.result().dataset("int1").selection().allGeom();
    model.result().dataset().remove("int1");
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result().numerical().create("pev1", "EvalPoint");
    model.result().numerical("pev1").set("expr", "poeq.g_u");
    model.result().numerical("pev1").set("descr", "Boundary flux/source");
    model.result().numerical().create("int2", "IntLine");
    model.result().numerical("int2").set("expr", "poeq.g_u");
    model.result().numerical("int2").set("descr", "Boundary flux/source");
    model.result().numerical("int2").selection().set(new int[]{5, 6, 7, 8});
    model.result().table().create("tbl3", "Table");
    model.result().table("tbl3").comments("Line Integration 2 (poeq.g_u)");
    model.result().numerical("int2").set("table", "tbl3");
    model.result().numerical("int2").setResult();
    model.result("pg1").run();
    model.result("pg1").run();

    model.geom("geom1").feature("c1").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c1").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c1").setIndex("pos", "(y_0*0.8+0.1)*L", 1);
    model.geom("geom1").run("c1");
    model.geom("geom1").feature().create("c2", "Circle");
    model.geom("geom1").feature("c2").set("base", "center");
    model.geom("geom1").feature("c2").set("pos", new String[]{"40", "140"});
    model.geom("geom1").feature("c2").set("r", "20");
    model.geom("geom1").run("c2");
    model.geom("geom1").feature().create("c3", "Circle");
    model.geom("geom1").feature("c3").set("base", "center");
    model.geom("geom1").feature("c3").set("pos", new String[]{"60", "120"});
    model.geom("geom1").feature("c3").set("r", "20");
    model.geom("geom1").run("c3");
    model.geom("geom1").feature().create("c4", "Circle");
    model.geom("geom1").feature("c4").set("base", "center");
    model.geom("geom1").feature("c4").set("pos", new String[]{"100", "140"});
    model.geom("geom1").feature("c4").set("r", "20");
    model.geom("geom1").run("c4");
    model.geom("geom1").feature().create("c5", "Circle");
    model.geom("geom1").feature("c5").set("base", "center");
    model.geom("geom1").feature("c5").set("pos", new String[]{"40", "80"});
    model.geom("geom1").feature("c5").set("r", "20");
    model.geom("geom1").run("c5");
    model.geom("geom1").feature().create("c6", "Circle");
    model.geom("geom1").feature("c6").set("base", "center");
    model.geom("geom1").feature("c6").set("pos", new String[]{"100", "100"});
    model.geom("geom1").feature("c6").set("r", "20");
    model.geom("geom1").run("c6");
    model.geom("geom1").feature().create("c7", "Circle");
    model.geom("geom1").feature("c7").set("base", "center");
    model.geom("geom1").feature("c7").set("pos", new String[]{"140", "100"});
    model.geom("geom1").feature("c7").set("r", "20");
    model.geom("geom1").run("c7");
    model.geom("geom1").feature().create("c8", "Circle");
    model.geom("geom1").feature("c8").set("base", "center");
    model.geom("geom1").feature("c8").set("pos", new String[]{"40", "180"});
    model.geom("geom1").feature("c8").set("r", "20");
    model.geom("geom1").run("c8");
    model.geom("geom1").feature().create("c9", "Circle");
    model.geom("geom1").feature("c9").set("base", "center");
    model.geom("geom1").feature("c9").set("pos", new String[]{"100", "60"});
    model.geom("geom1").feature("c9").set("r", "20");
    model.geom("geom1").run("c9");
    model.geom("geom1").feature().create("c10", "Circle");
    model.geom("geom1").feature("c10").set("base", "center");
    model.geom("geom1").feature("c10").set("pos", new String[]{"40", "80"});
    model.geom("geom1").feature("c10").set("r", "40");
    model.geom("geom1").run("c10");
    model.geom("geom1").feature().create("c11", "Circle");
    model.geom("geom1").feature("c11").set("base", "center");
    model.geom("geom1").feature("c11").set("pos", new String[]{"40", "20"});
    model.geom("geom1").feature("c11").set("r", "20");
    model.geom("geom1").run("c11");
    model.geom("geom1").feature().create("c12", "Circle");
    model.geom("geom1").feature("c12").set("base", "center");
    model.geom("geom1").feature("c12").set("pos", new String[]{"80", "160"});
    model.geom("geom1").feature("c12").set("r", "40");
    model.geom("geom1").run("c12");
    model.geom("geom1").feature().create("c13", "Circle");
    model.geom("geom1").feature("c13").set("base", "center");
    model.geom("geom1").feature("c13").set("pos", new String[]{"120", "160"});
    model.geom("geom1").feature("c13").set("r", "20");
    model.geom("geom1").run("c13");
    model.geom("geom1").feature().create("c14", "Circle");
    model.geom("geom1").feature("c14").set("base", "center");
    model.geom("geom1").feature("c14").set("pos", new String[]{"160", "140"});
    model.geom("geom1").feature("c14").set("r", "20");
    model.geom("geom1").run("c14");
    model.geom("geom1").feature().create("c15", "Circle");
    model.geom("geom1").feature("c15").set("base", "center");
    model.geom("geom1").feature("c15").set("pos", new String[]{"140", "60"});
    model.geom("geom1").feature("c15").set("r", "20");
    model.geom("geom1").run("c15");
    model.geom("geom1").feature().create("c16", "Circle");
    model.geom("geom1").feature("c16").set("base", "center");
    model.geom("geom1").feature("c16").set("pos", new String[]{"140", "40"});
    model.geom("geom1").feature("c16").set("r", "20");
    model.geom("geom1").run("c16");
    model.geom("geom1").feature().create("c17", "Circle");
    model.geom("geom1").feature("c17").set("base", "center");
    model.geom("geom1").feature("c17").set("pos", new String[]{"180", "80"});
    model.geom("geom1").feature("c17").set("r", "20");
    model.geom("geom1").run("c17");
    model.geom("geom1").feature("c2").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c3").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c4").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c5").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c6").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c2").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c3").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c4").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c5").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c6").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c7").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c7").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c8").setIndex("pos", "v", 0);
    model.geom("geom1").feature("c8").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c8").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c9").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c9").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c10").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c10").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c11").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c11").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c12").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c12").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c13").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c13").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c14").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c14").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c15").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c15").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c16").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c16").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c17").setIndex("pos", "(x_0*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c17").setIndex("pos", "(x_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c1").setIndex("pos", "(y_1*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c1").setIndex("pos", "(y_0*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c2").setIndex("pos", "(x_1*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c2").setIndex("pos", "(y_1*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c3").setIndex("pos", "(x_2*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c3").setIndex("pos", "(y_2*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c4").setIndex("pos", "(x_3*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c4").setIndex("pos", "(y_3*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c5").setIndex("pos", "(x_4*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c5").setIndex("pos", "(y_4*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c6").setIndex("pos", "(x_5*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c6").setIndex("pos", "(y_5*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c7").setIndex("pos", "(x_6*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c7").setIndex("pos", "(y_6*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c8").setIndex("pos", "(x_8*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c8").setIndex("pos", "(y_8*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c8").setIndex("pos", "(x_7*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c8").setIndex("pos", "(y_7*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c9").setIndex("pos", "(x_8*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c9").setIndex("pos", "(y_8*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c10").setIndex("pos", "(x_9*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c10").setIndex("pos", "(y_9*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c11").setIndex("pos", "(x_10*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c11").setIndex("pos", "(y_10*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c12").setIndex("pos", "(x_11*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c12").setIndex("pos", "(y_11*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c13").setIndex("pos", "(x_12*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c13").setIndex("pos", "(y_12*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c14").setIndex("pos", "(x_13*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c14").setIndex("pos", "(y_13*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c15").setIndex("pos", "(x_14*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c15").setIndex("pos", "(y_14*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c16").setIndex("pos", "(x_15*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c16").setIndex("pos", "(y_15*0.8+0.1)*L", 1);
    model.geom("geom1").feature("c17").setIndex("pos", "(x_16*0.8+0.1)*L", 0);
    model.geom("geom1").feature("c17").setIndex("pos", "(y_16*0.8+0.1)*L", 1);
    model.geom("geom1").runAll();
    model.geom("geom1").feature("c2").set("r", "R");
    model.geom("geom1").feature("c3").set("r", "R");
    model.geom("geom1").feature("c4").set("r", "R");
    model.geom("geom1").feature("c5").set("r", "R");
    model.geom("geom1").feature("c6").set("r", "R");
    model.geom("geom1").feature("c7").set("r", "R");
    model.geom("geom1").feature("c8").set("r", "R");
    model.geom("geom1").feature("c9").set("r", "R");
    model.geom("geom1").feature("c10").set("r", "R");
    model.geom("geom1").feature("c11").set("r", "R");
    model.geom("geom1").feature("c12").set("r", "R");
    model.geom("geom1").feature("c13").set("r", "R");
    model.geom("geom1").feature("c14").set("r", "R");
    model.geom("geom1").feature("c15").set("r", "R");
    model.geom("geom1").feature("c16").set("r", "R");
    model.geom("geom1").feature("c17").set("r", "R");
    model.geom("geom1").runAll();
    model.geom("geom1").feature().remove("dif1");
    model.geom("geom1").runAll();
    model.geom("geom1").run("c17");
    model.geom("geom1").feature().create("dif1", "Difference");
    model.geom("geom1").feature("dif1").selection("input")
         .set(new String[]{"sq1"});
    model.geom("geom1").feature("dif1").selection("input2")
         .set(new String[]{"c1", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"});
    model.geom("geom1").run("dif1");
    model.geom("geom1").run();
    model.geom("geom1").feature("dif1").active(false);
    model.geom("geom1").runAll();
    model.geom("geom1").feature("dif1").active(true);
    model.geom("geom1").runAll();
    model.geom("geom1").run();

    model.physics("poeq").feature("flux1").selection()
         .set(new int[]{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72});
    model.physics("poeq").feature().create("flux2", "FluxBoundary", 1);
    model.physics("poeq").feature().remove("flux1");
    model.physics("poeq").feature().create("flux3", "FluxBoundary", 1);
    model.physics("poeq").feature().create("flux4", "FluxBoundary", 1);
    model.physics("poeq").feature().remove("flux4");
    model.physics("poeq").feature("flux2").selection()
         .set(new int[]{7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 33, 34, 35, 36, 39, 40, 43, 44, 45, 46, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68});
    model.physics("poeq").feature("flux3").selection()
         .set(new int[]{5, 6, 9, 10, 27, 28, 31, 32, 37, 38, 41, 42, 47, 48, 51, 52, 69, 70, 71, 72});
    model.physics("poeq").feature("flux3").set("q", 1, "K_com");
    model.physics("poeq").feature("flux2").set("g", 1, "K_com");
    model.physics("poeq").feature("flux2").set("q", 1, "K_com");

    model.mesh("mesh1").run();

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.physics("poeq").feature("dir1").selection().set(new int[]{1, 4});

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.physics("poeq").feature("dir1").set("r", 1, "0.3");

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result().numerical().remove("gev1");
    model.result().numerical().remove("int1");
    model.result().numerical().remove("pev1");
    model.result().numerical().remove("int2");
    model.result().numerical().create("int1", "IntLine");
    model.result().numerical("int1").set("expr", "poeq.g_u");
    model.result().numerical("int1").set("descr", "Boundary flux/source");
    model.result().numerical().create("int2", "IntLine");
    model.result().numerical("int2").set("expr", "poeq.g_u");
    model.result().numerical("int2").set("descr", "Boundary flux/source");
    model.result().numerical("int2").selection()
         .set(new int[]{19, 20, 23, 24});
    model.result().numerical().create("int3", "IntLine");
    model.result().numerical("int3").selection()
         .set(new int[]{47, 48, 51, 52});
    model.result().numerical().create("int4", "IntLine");
    model.result().numerical("int4").selection()
         .set(new int[]{17, 18, 21, 22});
    model.result().numerical().create("int5", "IntLine");
    model.result().numerical("int5").selection()
         .set(new int[]{39, 40, 43, 44});
    model.result().numerical().create("int6", "IntLine");
    model.result().numerical("int6").selection()
         .set(new int[]{69, 70, 71, 72});
    model.result().numerical().create("int7", "IntLine");
    model.result().numerical("int7").selection()
         .set(new int[]{65, 66, 67, 68});
    model.result().numerical().create("int8", "IntLine");
    model.result().numerical("int8").selection()
         .set(new int[]{25, 26, 29, 30});
    model.result().numerical().create("int9", "IntLine");
    model.result().numerical("int9").selection()
         .set(new int[]{37, 38, 41, 42});
    model.result().numerical().create("int10", "IntLine");
    model.result().numerical("int10").selection()
         .set(new int[]{45, 46, 49, 50});
    model.result().numerical().create("int11", "IntLine");
    model.result().numerical("int11").selection()
         .set(new int[]{53, 54, 55, 56});
    model.result().numerical().create("int12", "IntLine");
    model.result().numerical("int12").selection().set(new int[]{5, 6, 9, 10});
    model.result().numerical().create("int13", "IntLine");
    model.result().numerical("int13").selection()
         .set(new int[]{7, 8, 11, 12});
    model.result().numerical().create("int14", "IntLine");
    model.result().numerical("int14").selection()
         .set(new int[]{13, 14, 15, 16});
    model.result().numerical().create("int15", "IntLine");
    model.result().numerical("int15").selection()
         .set(new int[]{27, 28, 31, 32});
    model.result().numerical().create("int16", "IntLine");
    model.result().numerical("int16").selection()
         .set(new int[]{57, 58, 59, 60});
    model.result().numerical().create("int17", "IntLine");
    model.result().numerical("int17").selection()
         .set(new int[]{61, 62, 63, 64});
    model.result().numerical("int1").selection()
         .set(new int[]{33, 34, 35, 36});
    model.result().numerical("int3").set("expr", "poeq.g_u");
    model.result().numerical("int3").set("descr", "Boundary flux/source");
    model.result().numerical("int4").set("expr", "poeq.g_u");
    model.result().numerical("int4").set("descr", "Boundary flux/source");
    model.result().numerical("int5").set("expr", "poeq.g_u");
    model.result().numerical("int5").set("descr", "Boundary flux/source");
    model.result().numerical("int6").set("expr", "poeq.g_u");
    model.result().numerical("int6").set("descr", "Boundary flux/source");
    model.result().numerical("int7").set("expr", "poeq.g_u");
    model.result().numerical("int7").set("descr", "Boundary flux/source");
    model.result().numerical("int9").set("expr", "poeq.g_u");
    model.result().numerical("int9").set("descr", "Boundary flux/source");
    model.result().numerical("int10").set("expr", "poeq.g_u");
    model.result().numerical("int10").set("descr", "Boundary flux/source");
    model.result().numerical("int11").set("expr", "poeq.g_u");
    model.result().numerical("int11").set("descr", "Boundary flux/source");
    model.result().numerical("int12").set("expr", "poeq.g_u");
    model.result().numerical("int12").set("descr", "Boundary flux/source");
    model.result().numerical("int13").set("expr", "poeq.g_u");
    model.result().numerical("int13").set("descr", "Boundary flux/source");
    model.result().numerical("int14").set("expr", "poeq.g_u");
    model.result().numerical("int14").set("descr", "Boundary flux/source");
    model.result().numerical("int15").set("expr", "poeq.g_u");
    model.result().numerical("int15").set("descr", "Boundary flux/source");
    model.result().numerical("int16").set("expr", "poeq.g_u");
    model.result().numerical("int16").set("descr", "Boundary flux/source");
    model.result().numerical("int17").set("expr", "poeq.g_u");
    model.result().numerical("int17").set("descr", "Boundary flux/source");

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "stat");
    model.sol("sol1").feature().create("v1", "Variables");
    model.sol("sol1").feature().create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result().table().remove("tbl1");
    model.result().table().remove("tbl2");
    model.result().table().remove("tbl3");
    model.result().table().create("tbl1", "Table");
    model.result().table("tbl1").comments("Line Integration 1 (poeq.g_u)");
    model.result().numerical("int1").set("table", "tbl1");
    model.result().numerical("int1").setResult();
    model.result().numerical("int2").set("table", "tbl1");
    model.result().numerical("int2").appendResult();
    model.result().numerical("int3").set("table", "tbl1");
    model.result().numerical("int3").appendResult();
    model.result().numerical("int4").set("table", "tbl1");
    model.result().numerical("int4").appendResult();
    model.result().numerical("int5").set("table", "tbl1");
    model.result().numerical("int5").appendResult();
    model.result().numerical("int6").set("table", "tbl1");
    model.result().numerical("int6").appendResult();
    model.result().numerical("int7").set("table", "tbl1");
    model.result().numerical("int7").appendResult();
    model.result().numerical("int8").set("table", "tbl1");
    model.result().numerical("int8").appendResult();
    model.result().numerical("int9").set("table", "tbl1");
    model.result().numerical("int9").appendResult();
    model.result().numerical("int10").set("table", "tbl1");
    model.result().numerical("int10").appendResult();
    model.result().numerical("int11").set("table", "tbl1");
    model.result().numerical("int11").appendResult();
    model.result().numerical("int12").set("table", "tbl1");
    model.result().numerical("int12").appendResult();
    model.result().numerical("int13").set("table", "tbl1");
    model.result().numerical("int13").appendResult();
    model.result().numerical("int14").set("table", "tbl1");
    model.result().numerical("int14").appendResult();
    model.result().numerical("int15").set("table", "tbl1");
    model.result().numerical("int15").appendResult();
    model.result().numerical("int16").set("table", "tbl1");
    model.result().numerical("int16").appendResult();
    model.result().numerical("int17").set("table", "tbl1");
    model.result().numerical("int17").appendResult();
    model.result().table("tbl1").name("q_linear");
    model.result().table("tbl1")
         .save("/home/pdavid/Bureau/SS/2D_cartesian/Updated_BCs/Figures_and_Tests/Multiple_sources/COMSOL_output/linear/q.txt");
    model.result("pg1").run();
    model.result().export().create("plot1", "pg1", "surf1", "Plot");
    model.result().export("plot1").set("header", "off");
    model.result().export("plot1")
         .set("filename", "/home/pdavid/Bureau/SS/2D_cartesian/Updated_BCs/Figures_and_Tests/Multiple_sources/COMSOL_output/linear/contour.txt");
    model.result("pg1").run();
    model.result().export().remove("plot1");
    model.result("pg1").run();
    model.result("pg1").set("windowtitle", "Graphics");
    model.result("pg1").set("window", "window1");
    model.result("pg1").set("windowtitle", "Plot 1");
    model.result("pg1").run();
    model.result("pg1").set("window", "graphics");
    model.result("pg1").set("windowtitle", "Graphics");
    model.result().export().create("plot1", "pg1", "surf1", "Plot");
    model.result().export("plot1").set("header", "off");
    model.result().export("plot1")
         .set("filename", "/home/pdavid/Bureau/SS/2D_cartesian/Updated_BCs/Figures_and_Tests/Multiple_sources/COMSOL_output/linear/contour.txt");
    model.result().export("plot1").run();
    model.result().numerical("int8").set("expr", "poeq.g_u");
    model.result().numerical("int8").set("descr", "Boundary flux/source");
    model.result().table().remove("tbl1");
    model.result().table().create("tbl1", "Table");
    model.result().table("tbl1").comments("Line Integration 1 (poeq.g_u)");
    model.result().numerical("int1").set("table", "tbl1");
    model.result().numerical("int1").setResult();
    model.result().table().create("tbl2", "Table");
    model.result().table("tbl2").comments("Line Integration 2 (poeq.g_u)");
    model.result().numerical("int2").set("table", "tbl2");
    model.result().numerical("int2").setResult();
    model.result().table().create("tbl3", "Table");
    model.result().table("tbl3").comments("Line Integration 3 (poeq.g_u)");
    model.result().numerical("int3").set("table", "tbl3");
    model.result().numerical("int3").setResult();
    model.result().table().create("tbl4", "Table");
    model.result().table("tbl4").comments("Line Integration 4 (poeq.g_u)");
    model.result().numerical("int4").set("table", "tbl4");
    model.result().numerical("int4").setResult();
    model.result().table().create("tbl5", "Table");
    model.result().table("tbl5").comments("Line Integration 5 (poeq.g_u)");
    model.result().numerical("int5").set("table", "tbl5");
    model.result().numerical("int5").setResult();
    model.result().table().create("tbl6", "Table");
    model.result().table("tbl6").comments("Line Integration 6 (poeq.g_u)");
    model.result().numerical("int6").set("table", "tbl6");
    model.result().numerical("int6").setResult();
    model.result().table().create("tbl7", "Table");
    model.result().table("tbl7").comments("Line Integration 7 (poeq.g_u)");
    model.result().numerical("int7").set("table", "tbl7");
    model.result().numerical("int7").setResult();
    model.result().table().create("tbl8", "Table");
    model.result().table("tbl8").comments("Line Integration 8 (poeq.g_u)");
    model.result().numerical("int8").set("table", "tbl8");
    model.result().numerical("int8").setResult();
    model.result().table().create("tbl9", "Table");
    model.result().table("tbl9").comments("Line Integration 9 (poeq.g_u)");
    model.result().numerical("int9").set("table", "tbl9");
    model.result().numerical("int9").setResult();
    model.result().table().create("tbl10", "Table");
    model.result().table("tbl10").comments("Line Integration 10 (poeq.g_u)");
    model.result().numerical("int10").set("table", "tbl10");
    model.result().numerical("int10").setResult();
    model.result().table().create("tbl11", "Table");
    model.result().table("tbl11").comments("Line Integration 11 (poeq.g_u)");
    model.result().numerical("int11").set("table", "tbl11");
    model.result().numerical("int11").setResult();
    model.result().table().create("tbl12", "Table");
    model.result().table("tbl12").comments("Line Integration 12 (poeq.g_u)");
    model.result().numerical("int12").set("table", "tbl12");
    model.result().numerical("int12").setResult();
    model.result().table().create("tbl13", "Table");
    model.result().table("tbl13").comments("Line Integration 13 (poeq.g_u)");
    model.result().numerical("int13").set("table", "tbl13");
    model.result().numerical("int13").setResult();
    model.result().table().create("tbl14", "Table");
    model.result().table("tbl14").comments("Line Integration 14 (poeq.g_u)");
    model.result().numerical("int14").set("table", "tbl14");
    model.result().numerical("int14").setResult();
    model.result().table().create("tbl15", "Table");
    model.result().table("tbl15").comments("Line Integration 15 (poeq.g_u)");
    model.result().numerical("int15").set("table", "tbl15");
    model.result().numerical("int15").setResult();
    model.result().table().create("tbl16", "Table");
    model.result().table("tbl16").comments("Line Integration 16 (poeq.g_u)");
    model.result().numerical("int16").set("table", "tbl16");
    model.result().numerical("int16").setResult();
    model.result().table().create("tbl17", "Table");
    model.result().table("tbl17").comments("Line Integration 17 (poeq.g_u)");
    model.result().numerical("int17").set("table", "tbl17");
    model.result().numerical("int17").setResult();
    model.result().table().remove("tbl1");
    model.result().table().remove("tbl17");
    model.result().table().remove("tbl3");
    model.result().table().remove("tbl2");
    model.result().table().remove("tbl4");
    model.result().table().remove("tbl8");
    model.result().table().remove("tbl5");
    model.result().table().remove("tbl6");
    model.result().table().remove("tbl7");
    model.result().table().remove("tbl9");
    model.result().table().remove("tbl10");
    model.result().table().remove("tbl11");
    model.result().table().remove("tbl13");
    model.result().table().remove("tbl12");
    model.result().table().remove("tbl15");
    model.result().table().remove("tbl16");
    model.result("pg1").run();
    model.result().table().remove("tbl14");
    model.result().table().create("tbl1", "Table");
    model.result().table("tbl1").comments("Line Integration 1 (poeq.g_u)");
    model.result().numerical("int1").set("table", "tbl1");
    model.result().numerical("int1").setResult();
    model.result().numerical("int2").set("table", "tbl1");
    model.result().numerical("int2").appendResult();
    model.result().numerical("int3").set("table", "tbl1");
    model.result().numerical("int3").appendResult();
    model.result().numerical("int4").set("table", "tbl1");
    model.result().numerical("int4").appendResult();
    model.result().numerical("int5").set("table", "tbl1");
    model.result().numerical("int5").appendResult();
    model.result().numerical("int6").set("table", "tbl1");
    model.result().numerical("int6").appendResult();
    model.result().numerical("int7").set("table", "tbl1");
    model.result().numerical("int7").appendResult();
    model.result().numerical("int8").set("table", "tbl1");
    model.result().numerical("int8").appendResult();
    model.result().numerical("int9").set("table", "tbl1");
    model.result().numerical("int9").appendResult();
    model.result().numerical("int10").set("table", "tbl1");
    model.result().numerical("int10").appendResult();
    model.result().numerical("int11").set("table", "tbl1");
    model.result().numerical("int11").appendResult();
    model.result().numerical("int12").set("table", "tbl1");
    model.result().numerical("int12").appendResult();
    model.result().numerical("int13").set("table", "tbl1");
    model.result().numerical("int13").appendResult();
    model.result().numerical("int14").set("table", "tbl1");
    model.result().numerical("int14").appendResult();
    model.result().numerical("int15").set("table", "tbl1");
    model.result().numerical("int15").appendResult();
    model.result().numerical("int16").set("table", "tbl1");
    model.result().numerical("int16").appendResult();
    model.result().numerical("int17").set("table", "tbl1");
    model.result().numerical("int17").appendResult();

    return model;
  }

}
