#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# In general, doing
#  run.pl some.log a b c is like running the command a b c in
# the bash shell, and putting the standard error and output into some.log.
# To run parallel jobs (backgrounded on the host machine), you can do (e.g.)
#  run.pl JOB=1:4 some.JOB.log a b c JOB is like running the command a b c JOB
# and putting it in some.JOB.log, for each one. [Note: JOB can be any identifier].
# If any of the jobs fails, this script will fail.

# A typical example is:
#  run.pl some.log my-prog "--opt=foo bar" foo \|  other-prog baz
# and run.pl will run something like:
# ( my-prog '--opt=foo bar' foo |  other-prog baz ) >& some.log
#
# Basically it takes the command-line arguments, quotes them
# as necessary to preserve spaces, and evaluates them with bash.
# In addition it puts the command line at the top of the log, and
# the start and end times of the command at the beginning and end.
# The reason why this is useful is so that we can create a different
# version of this program that uses a queueing system instead.

#use Data::Dumper;

@ARGV < 2 && die "usage: run.pl log-file command-line arguments...";

#print STDERR "COMMAND-LINE: " .  Dumper(\@ARGV) . "\n";
$job_pick = 'all';
$max_jobs_run = -1;
$jobstart = 1;
$jobend = 1;
$ignored_opts = ""; # These will be ignored.

# First parse an option like JOB=1:4, and any
# options that would normally be given to
# queue.pl, which we will just discard.

for (my $x = 1; $x <= 2; $x++) { # This for-loop is to
  # allow the JOB=1:n option to be interleaved with the
  # options to qsub.
  while (@ARGV >= 2 && $ARGV[0] =~ m:^-:) {
    # parse any options that would normally go to qsub, but which will be ignored here.
    my $switch = shift @ARGV;
    if ($switch eq "-V") {
      $ignored_opts .= "-V ";
    } elsif ($switch eq "--max-jobs-run" || $switch eq "-tc") {
      # we do support the option --max-jobs-run n, and its GridEngine form -tc n.
      # if the command appears multiple times uses the smallest option.
      if ( $max_jobs_run <= 0 ) {
          $max_jobs_run =  shift @ARGV;
      } else {
        my $new_constraint = shift @ARGV;
        if ( ($new_constraint < $max_jobs_run) ) {
          $max_jobs_run = $new_constraint;
        }
      }
      
      if (! ($max_jobs_run > 0)) {
        die "run.pl: invalid option --max-jobs-run $max_jobs_run";
      }
    } else {
      my $argument = shift @ARGV;
      if ($argument =~ m/^--/) {
        print STDERR "run.pl: WARNING: suspicious argument '$argument' to $switch; starts with '-'\n";
      }
      if ($switch eq "-sync" && $argument =~ m/^[yY]/) {
        $ignored_opts .= "-sync "; # Note: in the
        # corresponding code in queue.pl it says instead, just "$sync = 1;".
      } elsif ($switch eq "-pe") { # e.g. -pe smp 5
        my $argument2 = shift @ARGV;
        $ignored_opts .= "$switch $argument $argument2 ";
      } elsif ($switch eq "--gpu") {
        $using_gpu = $argument;
      } elsif ($switch eq "--pick") {
        if($argument =~ m/^(all|failed|incomplete)$/) {
          $job_pick = $argument;
        } else {
          print STDERR "run.pl: ERROR: --pick argument must be one of 'all', 'failed' or 'incomplete'"
        }
      } else {
        # Ignore option.
        $ignored_opts .= "$switch $argument ";
      }
    }
  }
  if ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+):(\d+)$/) { # e.g. JOB=1:20
    $jobname = $1;
    $jobstart = $2;
    $jobend = $3;
    if ($jobstart > $jobend) {
      die "run.pl: invalid job range $ARGV[0]";
    }
    if ($jobstart <= 0) {
      die "run.pl: invalid job range $ARGV[0], start must be strictly positive (this is required for GridEngine compatibility).";
    }
    shift;
  } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
    $jobname = $1;
    $jobstart = $2;
    $jobend = $2;
    shift;
  } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
    print STDERR "run.pl: Warning: suspicious first argument to run.pl: $ARGV[0]\n";
  }
}

# Users found this message confusing so we are removing it.
# if ($ignored_opts ne "") {
#   print STDERR "run.pl: Warning: ignoring options \"$ignored_opts\"\n";
# }

if ($max_jobs_run == -1) { # If --max-jobs-run option not set,
                           # then work out the number of processors if possible,
                           # and set it based on that.
  $max_jobs_run = 0;
  if ($using_gpu) {
    if (open(P, "nvidia-smi -L |")) {
      $max_jobs_run++ while (<P>);
      close(P);
    }
    if ($max_jobs_run == 0) {
      $max_jobs_run = 1;
      print STDERR "run.pl: Warning: failed to detect number of GPUs from nvidia-smi, using ${max_jobs_run}\n";
    }
  } elsif (open(P, "</proc/cpuinfo")) {  # Linux
    while (<P>) { if (m/^processor/) { $max_jobs_run++; } }
    if ($max_jobs_run == 0) {
      print STDERR "run.pl: Warning: failed to detect any processors from /proc/cpuinfo\n";
      $max_jobs_run = 10;  # reasonable default.
    }
    close(P);
  } elsif (open(P, "sysctl -a |")) {  # BSD/Darwin
    while (<P>) {
      if (m/hw\.ncpu\s*[:=]\s*(\d+)/) { # hw.ncpu = 4, or hw.ncpu: 4
        $max_jobs_run = $1;
        last;
      }
    }
    close(P);
    if ($max_jobs_run == 0) {
      print STDERR "run.pl: Warning: failed to detect any processors from sysctl -a\n";
      $max_jobs_run = 10;  # reasonable default.
    }
  } else {
    # allow at most 32 jobs at once, on non-UNIX systems; change this code
    # if you need to change this default.
    $max_jobs_run = 32;
  }
  # The just-computed value of $max_jobs_run is just the number of processors
  # (or our best guess); and if it happens that the number of jobs we need to
  # run is just slightly above $max_jobs_run, it will make sense to increase
  # $max_jobs_run to equal the number of jobs, so we don't have a small number
  # of leftover jobs.
  $num_jobs = $jobend - $jobstart + 1;
  if (!$using_gpu &&
      $num_jobs > $max_jobs_run && $num_jobs < 1.4 * $max_jobs_run) {
    $max_jobs_run = $num_jobs;
  }
}

sub pick_or_exit {
  # pick_or_exit ( $logfile ) 
  # Invoked before each job is started helps to run jobs selectively.
  #
  # Given the name of the output logfile decides whether the job must be 
  # executed (by returning from the subroutine) or not (by terminating the
  # process calling exit)
  # 
  # PRE: $job_pick is a global variable set by command line switch --pick
  #      and indicates which class of jobs must be executed.
  #
  # 1) If a failed job is not executed the process exit code will indicate 
  #    failure, just as if the task was just executed  and failed.
  #
  # 2) If a task is incomplete it will be executed. Incomplete may be either
  #    a job whose log file does not contain the accounting notes in the end,
  #    or a job whose log file does not exist.
  #
  # 3) If the $job_pick is set to 'all' (default behavior) a task will be
  #    executed regardless of the result of previous attempts.
  #
  # This logic could have been implemented in the main execution loop
  # but a subroutine to preserve the current level of readability of
  # that part of the code.
  #
  # Alexandre Felipe, (o.alexandre.felipe@gmail.com) 14th of August of 2020
  #
  if($job_pick eq 'all'){
    return; # no need to bother with the previous log
  }
  open my $fh, "<", $_[0] or return; # job not executed yet
  my $log_line;
  my $cur_line;
  while ($cur_line = <$fh>) {
    if( $cur_line =~ m/# Ended \(code .*/ ) {
      $log_line = $cur_line;
    }
  }
  close $fh;
  if (! defined($log_line)){
    return; # incomplete
  }
  if ( $log_line =~ m/# Ended \(code 0\).*/ ) {
    exit(0); # complete
  } elsif ( $log_line =~ m/# Ended \(code \d+(; signal \d+)?\).*/ ){
    if ($job_pick !~ m/^(failed|all)$/) {
      exit(1); # failed but not going to run
    } else {
      return; # failed
    }
  } elsif ( $log_line =~ m/.*\S.*/ ) {
    return; # incomplete jobs are always run
  }
}


$logfile = shift @ARGV;

if (defined $jobname && $logfile !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "run.pl: you are trying to run a parallel job but "
    . "you are putting the output into just one log file ($logfile)\n";
  exit(1);
}

$cmd = "";

foreach $x (@ARGV) {
    if ($x =~ m/^\S+$/) { $cmd .=  $x . " "; }
    elsif ($x =~ m:\":) { $cmd .= "'$x' "; }
    else { $cmd .= "\"$x\" "; }
}

#$Data::Dumper::Indent=0;
$ret = 0;
$numfail = 0;
%active_pids=();

use POSIX ":sys_wait_h";
for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  if (scalar(keys %active_pids) >= $max_jobs_run) {

    # Lets wait for a change in any child's status
    # Then we have to work out which child finished
    $r = waitpid(-1, 0);
    $code = $?;
    if ($r < 0 ) { die "run.pl: Error waiting for child process"; } # should never happen.
    if ( defined $active_pids{$r} ) {
        $jid=$active_pids{$r};
        $fail[$jid]=$code;
        if ($code !=0) { $numfail++;}
        delete $active_pids{$r};
        # print STDERR "Finished: $r/$jid " .  Dumper(\%active_pids) . "\n";
    } else {
        die "run.pl: Cannot find the PID of the child process that just finished.";
    }

    # In theory we could do a non-blocking waitpid over all jobs running just
    # to find out if only one or more jobs finished during the previous waitpid()
    # However, we just omit this and will reap the next one in the next pass
    # through the for(;;) cycle
  }
  $childpid = fork();
  if (!defined $childpid) { die "run.pl: Error forking in run.pl (writing to $logfile)"; }
  if ($childpid == 0) { # We're in the child... this branch
    # executes the job and returns (possibly with an error status).
    if (defined $jobname) {
      $cmd =~ s/$jobname/$jobid/g;
      $logfile =~ s/$jobname/$jobid/g;
    }
    # exit if the job does not need to be executed
    pick_or_exit( $logfile );

    system("mkdir -p `dirname $logfile` 2>/dev/null");
    open(F, ">$logfile") || die "run.pl: Error opening log file $logfile";
    print F "# " . $cmd . "\n";
    print F "# Started at " . `date`;
    $starttime = `date +'%s'`;
    print F "#\n";
    close(F);

    # Pipe into bash.. make sure we're not using any other shell.
    open(B, "|bash") || die "run.pl: Error opening shell command";
    print B "( " . $cmd . ") 2>>$logfile >> $logfile";
    close(B);                   # If there was an error, exit status is in $?
    $ret = $?;

    $lowbits = $ret & 127;
    $highbits = $ret >> 8;
    if ($lowbits != 0) { $return_str = "code $highbits; signal $lowbits" }
    else { $return_str = "code $highbits"; }

    $endtime = `date +'%s'`;
    open(F, ">>$logfile") || die "run.pl: Error opening log file $logfile (again)";
    $enddate = `date`;
    chop $enddate;
    print F "# Accounting: time=" . ($endtime - $starttime) . " threads=1\n";
    print F "# Ended ($return_str) at " . $enddate . ", elapsed time " . ($endtime-$starttime) . " seconds\n";
    close(F);
    exit($ret == 0 ? 0 : 1);
  } else {
    $pid[$jobid] = $childpid;
    $active_pids{$childpid} = $jobid;
    # print STDERR "Queued: " .  Dumper(\%active_pids) . "\n";
  }
}

# Now we have submitted all the jobs, lets wait until all the jobs finish
foreach $child (keys %active_pids) {
    $jobid=$active_pids{$child};
    $r = waitpid($pid[$jobid], 0);
    $code = $?;
    if ($r == -1) { die "run.pl: Error waiting for child process"; } # should never happen.
    if ($r != 0) { $fail[$jobid]=$code; $numfail++ if $code!=0; } # Completed successfully
}

# Some sanity checks:
# The $fail array should not contain undefined codes
# The number of non-zeros in that array  should be equal to $numfail
# We cannot do foreach() here, as the JOB ids do not start at zero
$failed_jids=0;
for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $job_return = $fail[$jobid];
  if (not defined $job_return ) {
    # print Dumper(\@fail);

    die "run.pl: Sanity check failed: we have indication that some jobs are running " .
      "even after we waited for all jobs to finish" ;
  }
  if ($job_return != 0 ){ $failed_jids++;}
}
if ($failed_jids != $numfail) {
  die "run.pl: Sanity check failed: cannot find out how many jobs failed ($failed_jids x $numfail)."
}
if ($numfail > 0) { $ret = 1; }

if ($ret != 0) {
  $njobs = $jobend - $jobstart + 1;
  if ($njobs == 1) {
    if (defined $jobname) {
      $logfile =~ s/$jobname/$jobstart/; # only one numbered job, so replace name with
                                         # that job.
    }
    print STDERR "run.pl: job failed, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "run.pl: probably you forgot to put JOB=1:\$nj in your script.";
    }
  }
  else {
    $logfile =~ s/$jobname/*/g;
    print STDERR "run.pl: $numfail / $njobs failed, log is in $logfile\n";
  }
}


exit ($ret);
