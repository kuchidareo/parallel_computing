package org.apache.hadoop.examples;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class MatrixCalculation {

  public static class TokenizerMapper
      extends Mapper<Object, Text, IntWritable, DoubleWritable> {

    private IntWritable word = new IntWritable();
    private DoubleWritable one = new DoubleWritable();
    private double[] vector;

    // Read vector once per Mapper
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      String vectorString = context.getConfiguration().get("vector_values");
      String[] values = vectorString.split(",");
      vector = new double[values.length];

      for (int j = 0; j < values.length; j++) {
        vector[j] = Double.parseDouble(values[j].trim());
      }
    }

    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {

      String[] row = value.toString().split(",");
      int i = Integer.parseInt(row[0].trim());

      if (row.length != vector.length + 1) {
        throw new IOException("Matrix row length does not match vector length");
      }

      double sum = 0;
      for (int j = 0; j < vector.length; j++) {
        sum += Double.parseDouble(row[j + 1].trim()) * vector[j];
      }

      word.set(i);
      one.set(sum);
      context.write(word, one);
    }
  }

  public static class IntSumReducer
      extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {

    private DoubleWritable result = new DoubleWritable();

    public void reduce(IntWritable key, Iterable<DoubleWritable> values,
                       Context context) throws IOException, InterruptedException {
      double sum = 0;
      for (DoubleWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    org.apache.log4j.BasicConfigurator.configure();
    org.apache.log4j.Logger.getRootLogger().setLevel(org.apache.log4j.Level.INFO);
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

    if (otherArgs.length != 3) {
      System.err.println("Usage: MatrixCalculation <matrix> <vector> <out>");
      System.exit(2);
    }

    // Read the vector and pass it to the Mappers
    String vectorString;
    try (BufferedReader reader = new BufferedReader(new FileReader(otherArgs[1]))) {
      vectorString = reader.readLine();
    }
    if (vectorString == null || vectorString.trim().isEmpty()) {
      throw new IOException("Vector file is empty");
    }
    conf.set("vector_values", vectorString);

    Job job = Job.getInstance(conf, "matrix calculation");
    job.setJarByClass(MatrixCalculation.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(DoubleWritable.class);

    for (int i = 0; i < otherArgs.length - 2; ++i) {
      FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
    }

    FileOutputFormat.setOutputPath(job,
        new Path(otherArgs[otherArgs.length - 1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
