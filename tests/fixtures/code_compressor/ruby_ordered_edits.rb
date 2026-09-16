#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require_relative "processor"
load "extensions"
autoload :Helper, "helper"

class Worker
  include Enumerable
  CONST = "stable"
  public

  def process(items)
    first = items.map(&:strip)
    second = first.map(&:upcase)
    third = second.reverse
    fourth = third.join(",")
    fifth = fourth.strip
    sixth = fifth.reverse
    seventh = sixth.upcase
    eighth = seventh
    ninth = eighth
    tenth = ninth
    tenth
  end
end

describe Worker do
  def helper
    "kept opaque"
  end
end
